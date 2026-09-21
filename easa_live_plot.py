#!/usr/bin/env python3
"""
Live version of easa_plot.py: re-reads the CSV every second and updates
the plot while easa_standalone.py is still running and logging.

Usage (run in a SEPARATE terminal from easa_standalone.py):
    python easa_live_plot.py [csv_path]

Close the plot window to stop.
"""
import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from easa_plot import load_log, plot_motivations, plot_ethogram, CHANNEL_NAMES, find_latest_log

REFRESH_MS = 1000


def main():
    # DEVIATION FROM THE ORIGINAL, at the user's request: same reason as
    # in easa_plot.py -- there is no longer a fixed "easa_log.csv"; each run
    # has its own timestamp. By default, it uses the most recent log it finds;
    # a specific CSV path can still be provided manually.
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = find_latest_log()
        if csv_path is None:
            print("ERROR: no easa_log*.csv found next to this script, and no "
                  "path given. Usage: python easa_live_plot.py [csv_path]")
            sys.exit(1)
        print(f"[OK] No csv_path given, using the most recent log: {csv_path}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    def update(frame):
        if not os.path.exists(csv_path):
            return
        try:
            steps, fear, hunger, bg = load_log(csv_path)
        except Exception:
            return   # file mid-write, just skip this tick
        if len(steps) == 0:
            return

        for ax in axes:
            ax.clear()
        plot_motivations(axes[0], steps, fear, hunger)
        plot_ethogram(axes[1], steps, bg)

        # quick "what's winning right now" readout in the title
        last = {n: bg[n][-1] for n in CHANNEL_NAMES}
        winner = min(last, key=last.get)

        # NEW, at the user's request: also show can_pickup_stm live, so an
        # aborted-before-commit attempt (stm resets to 0 before reaching
        # COMMIT_STEP=8) is visible on screen while watching, not just
        # after the fact in the CSV.
        can_pickup_stm = "?"
        try:
            with open(csv_path) as f:
                last_row = None
                for last_row in csv.DictReader(f):
                    pass
            if last_row is not None:
                can_pickup_stm = last_row.get("can_pickup_stm", "?")
        except Exception:
            pass

        fig.suptitle(f"step {steps[-1]}  |  fear={fear[-1]:.2f}  hunger={hunger[-1]:.2f}  "
                     f"|  selected: {winner}  |  can_pickup_stm: {can_pickup_stm}", fontsize=11)
        plt.tight_layout()

    ani = animation.FuncAnimation(fig, update, interval=REFRESH_MS, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()