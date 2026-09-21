#!/usr/bin/env python3
"""
Reads the CSV logged by easa_standalone.py and produces:
  1) A motivations plot (fear, hunger over time) -- like the original's
     "Motivations" window.
  2) An ethogram (one row per channel, coloured by whether it's selected)
     -- like the original's "Behaviours" window. Reconstructed from
     bg_output using the SAME semantics established earlier: LOW output =
     selected/disinhibited.

Usage:
    python easa_plot.py [csv_path]
"""
import sys
import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

CHANNEL_NAMES = ["can_seek", "can_pickup", "wall_seek", "corner_seek", "corner_deposit"]
SELECTED_THRESHOLD = 0.3   # bg_output below this counts as "selected" for the ethogram


def find_latest_log():
    """DEVIATION FROM THE ORIGINAL, at the user's request: easa_standalone.py
    now adds a timestamp to the log filename for each run
    (so runs do not overwrite one another -- see the session notes), meaning
    there is no longer a fixed "easa_log.csv" to search for by default.
    Instead, it searches for all "easa_log*.csv" files (excluding _plot.png
    files and metadata .json files) in the script directory and returns the
    most recently modified one. A specific CSV path can still be provided on
    the command line to bypass this behavior."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [p for p in glob.glob(os.path.join(here, "easa_log*.csv"))]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_log(path):
    steps, fear, hunger = [], [], []
    bg = {n: [] for n in CHANNEL_NAMES}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            fear.append(float(row["fear"]))
            hunger.append(float(row["hunger"]))
            for n in CHANNEL_NAMES:
                bg[n].append(float(row[f"bg_{n}"]))
    return (np.array(steps), np.array(fear), np.array(hunger),
            {n: np.array(v) for n, v in bg.items()})


def plot_motivations(ax, steps, fear, hunger):
    ax.plot(steps, fear, color="tab:blue", label="fear")
    ax.plot(steps, hunger, color="tab:pink", label="hunger")
    ax.set_ylabel("level")
    ax.set_xlabel("decision step")
    ax.set_title("Motivations")
    ax.legend(loc="upper right", bbox_to_anchor=(-0.01, 1.0))
    ax.grid(alpha=0.3)


def plot_ethogram(ax, steps, bg):
    for i, n in enumerate(CHANNEL_NAMES):
        selected = bg[n] < SELECTED_THRESHOLD
        ax.fill_between(steps, i, i + 0.8, where=selected,
                         step="post", color="tab:red", alpha=0.85)
        ax.fill_between(steps, i, i + 0.8, where=~selected,
                         step="post", color="#eeeeee", alpha=0.85)
    ax.set_yticks([i + 0.4 for i in range(len(CHANNEL_NAMES))])
    ax.set_yticklabels(CHANNEL_NAMES)
    ax.set_xlabel("decision step")
    ax.set_title(f"Ethogram (red = selected, bg_output < {SELECTED_THRESHOLD})")


def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = find_latest_log()
        if csv_path is None:
            print("ERROR: no easa_log*.csv found next to this script, and no "
                  "path given. Usage: python easa_plot.py [csv_path]")
            sys.exit(1)
        print(f"[OK] No csv_path given, using the most recent log: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"ERROR: log not found at {csv_path}")
        sys.exit(1)

    steps, fear, hunger, bg = load_log(csv_path)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    plot_motivations(axes[0], steps, fear, hunger)
    plot_ethogram(axes[1], steps, bg)
    plt.tight_layout()

    out_path = os.path.splitext(csv_path)[0] + "_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved plot to {out_path}")


if __name__ == "__main__":
    main()