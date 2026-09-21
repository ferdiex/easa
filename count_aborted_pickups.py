"""
count_aborted_pickups.py

Reads can_pickup_stm from an existing easa_log_*.csv (no live watching
needed -- run this on a log AFTER the simulation finishes) and counts how
many can_pickup attempts aborted BEFORE committing (stm reset to 0 while
below COMMIT_STEP=8, i.e. the arm started moving but the attempt was
abandoned before the fixed-action-pattern point) vs. how many committed
(stm reached 8 or more before returning to 0, whether it finished
cleanly or not).

Usage:
    python3 count_aborted_pickups.py <path_to.csv>
    python3 count_aborted_pickups.py .    (scans every easa_log_*.csv here)
"""
import sys
import os
import glob
import csv

COMMIT_STEP = 8


def count_attempts(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if "can_pickup_stm" not in rows[0]:
        return None  # log predates this column

    aborted_before_commit = 0
    committed = 0
    peak_stm_this_episode = 0
    prev_stm = 0
    for row in rows:
        stm = int(row["can_pickup_stm"])
        peak_stm_this_episode = max(peak_stm_this_episode, stm)
        if prev_stm > 0 and stm == 0:
            # an episode just ended (reset to 0) -- classify by how far it got
            if peak_stm_this_episode < COMMIT_STEP:
                aborted_before_commit += 1
            else:
                committed += 1
            peak_stm_this_episode = 0
        prev_stm = stm

    return aborted_before_commit, committed


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 count_aborted_pickups.py <csv_or_directory>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        csv_paths = sorted(glob.glob(os.path.join(target, "easa_log_*.csv")))
    else:
        csv_paths = [target]

    total_aborted = 0
    total_committed = 0
    for csv_path in csv_paths:
        result = count_attempts(csv_path)
        if result is None:
            print(f"[skip] {csv_path}: no can_pickup_stm column")
            continue
        aborted, committed = result
        total_aborted += aborted
        total_committed += committed
        print(f"{os.path.basename(csv_path)}: aborted before commitment (stm<8) = {aborted}, "
              f"committed (stm>=8) = {committed}")

    print()
    total = total_aborted + total_committed
    if total > 0:
        print(f"TOTAL: {total_aborted}/{total} aborted before commitment "
              f"({100 * total_aborted / total:.0f}%), {total_committed}/{total} committed")
    else:
        print("No can_pickup attempts were found in the reviewed logs.")


if __name__ == "__main__":
    main()
