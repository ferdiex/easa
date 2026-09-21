"""
analyze_channel_time.py

Reads active_channel from an existing easa_log_*.csv (no live watching
needed) and reports what fraction of decision steps were spent in each
channel -- with special attention to "none" (no channel crossed
THALAMIC_TONIC that step), the cleanest, most direct signature of
indecision/dithering: not a wrong choice, no choice at all.

Usage:
    python3 analyze_channel_time.py <path_to.csv>
    python3 analyze_channel_time.py .    (scans every easa_log_*.csv here)
"""
import sys
import os
import glob
import csv
from collections import Counter


def channel_breakdown(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    counts = Counter(row["active_channel"] for row in rows)
    total = len(rows)
    return counts, total


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_channel_time.py <csv_or_directory>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        csv_paths = sorted(glob.glob(os.path.join(target, "easa_log_*.csv")))
    else:
        csv_paths = [target]

    grand_counts = Counter()
    grand_total = 0
    for csv_path in csv_paths:
        counts, total = channel_breakdown(csv_path)
        grand_counts.update(counts)
        grand_total += total
        none_pct = 100 * counts.get("none", 0) / total if total else 0
        print(f"{os.path.basename(csv_path)}: {total} rows, "
              f"'none' (nothing selected) = {counts.get('none', 0)} ({none_pct:.1f}%)")

    print()
    print("Total breakdown, all logs combined:")
    for channel, count in grand_counts.most_common():
        print(f"  {channel:16s} {count:6d}  ({100 * count / grand_total:5.1f}%)")


if __name__ == "__main__":
    main()