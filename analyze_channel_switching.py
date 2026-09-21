"""
analyze_channel_switching.py

The classic definition of "dithering" in action-selection literature
(including Prescott's own papers) is rapid, unproductive SWITCHING
between similarly-weighted behaviours -- not "no selection" (checked in
analyze_channel_time.py: 0% "none") and not "aborting mid-attempt"
(checked in count_aborted_pickups.py: 0 aborts). This script checks the
remaining, classic sense directly: how long does each consecutive run of
the same active_channel last, and how often does the channel actually
change? Many short runs = flickering/dithering. Few long runs = the
opposite -- one behaviour dominating and NOT letting go, a different
failure mode entirely.

Usage:
    python3 analyze_channel_switching.py <path_to.csv>
    python3 analyze_channel_switching.py .   (scans every easa_log_*.csv here)
"""
import sys
import os
import glob
import csv
from collections import defaultdict

import numpy as np


def run_lengths(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    channels = [row["active_channel"] for row in rows]

    runs = []   # (channel, length_in_logged_rows)
    prev = None
    length = 0
    for ch in channels:
        if ch == prev:
            length += 1
        else:
            if prev is not None:
                runs.append((prev, length))
            prev = ch
            length = 1
    if prev is not None:
        runs.append((prev, length))
    return runs, len(rows)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_channel_switching.py <csv_or_directory>")
        sys.exit(1)

    target = sys.argv[1]
    csv_paths = (sorted(glob.glob(os.path.join(target, "easa_log_*.csv")))
                 if os.path.isdir(target) else [target])

    for csv_path in csv_paths:
        runs, total_rows = run_lengths(csv_path)
        n_switches = len(runs) - 1 if runs else 0
        lengths = np.array([r[1] for r in runs])

        print(f"{os.path.basename(csv_path)}:")
        print(f"  {total_rows} logged rows, {len(runs)} segments, {n_switches} channel changes")
        print(f"  segment duration: median={np.median(lengths):.0f} rows, "
              f"mean={lengths.mean():.1f}, max={lengths.max()}")

        # Per channel, to determine whether ANY particular channel flickers
        # while another remains stuck -- a global average could hide this.
        by_channel = defaultdict(list)
        for ch, length in runs:
            by_channel[ch].append(length)
        for ch in sorted(by_channel, key=lambda c: -sum(by_channel[c])):
            lens = np.array(by_channel[ch])
            print(f"    {ch:16s} n_segments={len(lens):4d}  median={np.median(lens):6.0f}  "
                  f"max={lens.max():6d}  total_rows={lens.sum():6d}")
        print()


if __name__ == "__main__":
    main()