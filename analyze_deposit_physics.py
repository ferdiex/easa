"""
analyze_deposit_physics.py

Mines every easa_log_*.csv already produced by run_dopamine_sweep.sh (or
any other run of easa_standalone.py that used the canN_z/canN_tilt_deg
logging added this session) for what actually happens to a can right
after corner_deposit releases it. No new simulation runs are needed --
this only reads existing logs.

For every corner_deposit episode that ends with the gripper releasing
something, the released can is classified by its position/orientation at
the moment of release into one of four outcomes:
  - exited:          |x| or |y| beyond the arena half-extent (0.3m) --
                      a clean deposit, the can left the room.
  - stuck_on_wall:    still inside/near the bounds but sitting clearly
                      higher than its resting height (z > 0.03m, vs.
                      CAN_HEIGHT/2 = 0.02m at rest) -- the "expulsion by
                      the wall" artifact (session finding, WALL_HEIGHT <
                      CAN_HEIGHT).
  - fell_over_inside: still inside the arena, resting height, but tilted
                      more than 30 degrees from upright -- toppled
                      instead of ejected.
  - upright_inside:   released but still standing normally inside the
                      arena (didn't clear the wall, didn't fall over
                      either -- the release just didn't push it out).

Usage:
    python3 analyze_deposit_physics.py [directory]
Writes deposit_physics_summary.csv (one row per release event) and
prints an overall rate table to stdout. Also breaks the rates down by
dopamine value (read from the matching .json metadata) since a
plausible hypothesis is that dopamine-driven approach-speed/behaviour
changes affect how hard the can gets hit on the way in.
"""
import sys
import os
import glob
import json
import csv
import math
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARENA_HALF = 0.3       # WORLD_SIZE / 2, from easa_arena.py
CAN_RESTING_Z = 0.02    # CAN_HEIGHT / 2, from easa_arena.py
STUCK_Z_THRESHOLD = 0.03       # clearly above resting height
FALLEN_TILT_THRESHOLD_DEG = 30.0
N_CANS = 4
OUTCOMES = ["exited", "stuck_on_wall", "fell_over_inside", "upright_inside"]


def write_markdown_table(path, caption, headers, data_rows):
    """Writes a simple, clean markdown table (caption + header + rows).
    Markdown tables paste cleanly into most LaTeX pipelines (via pandoc)
    and into Word -- this is deliberately plain (no styling) so it is
    easy to convert to whatever the target journal's table format needs,
    rather than trying to guess that format here."""
    with open(path, "w") as f:
        f.write(f"{caption}\n\n")
        f.write("| " + " | ".join(str(h) for h in headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in data_rows:
            f.write("| " + " | ".join(str(v) for v in row) + " |\n")


def classify_release(x, y, z, tilt_deg):
    if abs(x) > ARENA_HALF or abs(y) > ARENA_HALF:
        return "exited"
    if z > STUCK_Z_THRESHOLD:
        return "stuck_on_wall"
    if tilt_deg > FALLEN_TILT_THRESHOLD_DEG:
        return "fell_over_inside"
    return "upright_inside"


def find_release_events(csv_path):
    """Walk one run's log and yield a dict per release event: which can
    was held (by proximity, same heuristic used throughout this session's
    manual analysis), and that can's x/y/z/tilt at the moment of release.

    FIXED, found while looking into why D=1.0 had zero detected releases:
    this used to only track "what's held" while active_channel=="corner_
    deposit" -- but the gate-blending finding from this session (see
    notes_sesion_easa_real_v2.md, "La pregunta huntingtoniana") showed
    that a behaviour can keep issuing real motor commands (including a
    release) without being the logged active_channel, especially at high
    dopamine. Requiring the channel name was blind exactly in that
    regime. Now tracks "held" from gripper_detected alone, regardless of
    which channel is logged as active."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    events = []
    prev_grip = None
    held_can_idx = None

    for i, row in enumerate(rows):
        grip = row["gripper_detected"]

        if grip == "1":
            # holding something -- track which can this is, by proximity,
            # regardless of which channel is logged as active (see note
            # above).
            rx, ry = float(row["robot_x"]), float(row["robot_y"])
            best = None
            for ci in range(N_CANS):
                cx, cy = float(row[f"can{ci}_x"]), float(row[f"can{ci}_y"])
                d = math.hypot(cx - rx, cy - ry)
                if best is None or d < best[0]:
                    best = (d, ci)
            if best is not None and best[0] < 0.08:  # HELD_CAN_RELEASE_DIST
                held_can_idx = best[1]

        # A release: was holding last row, not holding now, AND the
        # robot's own corner_detected sensor was reading true at (or
        # near) that moment -- this is what actually distinguishes a
        # genuine deposit attempt from an accidental mid-transport drop
        # (which COMMIT_STEP already made rare, but not zero). Using the
        # robot's real sensor reading here instead of a distance-to-an-
        # assumed-corner-coordinate heuristic, since DEPOSIT_CORNER is
        # just a reference point for that sensor's own detection zone,
        # not necessarily where the robot physically ends up sitting --
        # an earlier version of this script guessed a fixed radius
        # around that coordinate and it was too strict, throwing out
        # genuine deposits. Checks a small window of rows around the
        # release (not just the exact release row) since the log is
        # sampled every N decision steps and the sensor could read
        # true a step or two before/after the row that happens to be
        # logged.
        if prev_grip == "1" and grip == "0" and held_can_idx is not None:
            window = rows[max(0, i - 2):i + 1]
            corner_was_detected = any(w["corner_detected"] == "1" for w in window)
            if corner_was_detected:
                x = float(row[f"can{held_can_idx}_x"])
                y = float(row[f"can{held_can_idx}_y"])
                z = float(row[f"can{held_can_idx}_z"])
                tilt = float(row[f"can{held_can_idx}_tilt_deg"])
                events.append({
                    "step": row["step"],
                    "can_index": held_can_idx,
                    "x": x, "y": y, "z": z, "tilt_deg": tilt,
                    "outcome": classify_release(x, y, z, tilt),
                })
            held_can_idx = None  # reset either way, ready for the next episode

        prev_grip = grip

    return events


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    csv_paths = sorted(glob.glob(os.path.join(directory, "easa_log_*.csv")))
    if not csv_paths:
        print(f"No easa_log_*.csv files found in {directory}")
        sys.exit(1)

    all_rows = []
    for csv_path in csv_paths:
        json_path = csv_path.replace(".csv", ".json")
        dopamine = None
        if os.path.exists(json_path):
            with open(json_path) as f:
                dopamine = json.load(f).get("dopamine_d1")

        try:
            events = find_release_events(csv_path)
        except KeyError as e:
            print(f"[skip] {csv_path}: missing column {e} "
                  f"(log predates the canN_z/canN_tilt_deg addition?)")
            continue

        for ev in events:
            ev["csv_path"] = os.path.basename(csv_path)
            ev["dopamine"] = dopamine
            all_rows.append(ev)

    if not all_rows:
        print("No corner_deposit release events found in any log.")
        sys.exit(0)

    with open(os.path.join(directory, "deposit_physics_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[OK] {len(all_rows)} release events across {len(csv_paths)} logs "
          f"-> deposit_physics_summary.csv\n")

    # Overall rates.
    outcomes = [r["outcome"] for r in all_rows]
    n = len(outcomes)
    print("Overall (all dopamine values pooled):")
    for outcome in ["exited", "stuck_on_wall", "fell_over_inside", "upright_inside"]:
        count = outcomes.count(outcome)
        print(f"  {outcome:18s} {count:5d} / {n} ({100 * count / n:5.1f}%)")

    # Broken down by dopamine, in case approach dynamics matter.
    by_dopamine = defaultdict(list)
    for r in all_rows:
        by_dopamine[r["dopamine"]].append(r["outcome"])

    if len(by_dopamine) > 1:
        print("\nBy dopamine value:")
        for dopamine in sorted(by_dopamine, key=lambda d: (d is None, d)):
            group = by_dopamine[dopamine]
            n_group = len(group)
            counts = {o: group.count(o) for o in
                      ["exited", "stuck_on_wall", "fell_over_inside", "upright_inside"]}
            pct = {o: 100 * c / n_group for o, c in counts.items()}
            label = f"D={dopamine}" if dopamine is not None else \
                "D=UNKNOWN (no dopamine_d1 in .json -- likely a log from before " \
                "this metadata field existed, or outside run_dopamine_sweep.sh; " \
                "re-run in a clean, dedicated directory to avoid this)"
            print(f"  {label} (n={n_group}): "
                  + ", ".join(f"{o}={pct[o]:.0f}%" for o in counts))

    # NEW: Figure 4 (continuing the numbering from aggregate_sweep_
    # results.py's Figures 1-3) -- a GROUPED bar chart, not stacked, per
    # the user's request: stacked-to-100% bars make it hard to compare
    # one outcome (e.g. "exited") across dopamine values, since each
    # bar's baseline shifts. Grouped bars keep every outcome's own
    # baseline at zero, at the cost of using more horizontal space --
    # an acceptable trade for a single-column figure.
    known_dopamines = sorted(d for d in by_dopamine if d is not None)
    if known_dopamines:
        x = np.arange(len(known_dopamines))
        width = 0.2
        fig4, ax4 = plt.subplots(figsize=(7, 4.5))
        colors = {"exited": "tab:green", "stuck_on_wall": "tab:orange",
                  "fell_over_inside": "tab:red", "upright_inside": "tab:blue"}
        for i, outcome in enumerate(OUTCOMES):
            pct_by_dop = [100 * by_dopamine[d].count(outcome) / len(by_dopamine[d])
                          for d in known_dopamines]
            ax4.bar(x + (i - 1.5) * width, pct_by_dop, width,
                    label=outcome.replace("_", " "), color=colors[outcome])
        ax4.set_xticks(x)
        ax4.set_xticklabels([str(d) for d in known_dopamines])
        ax4.set_xlabel("dopamine (D1 = D2)")
        ax4.set_ylabel("% of release events")
        ax4.set_title("Deposit-release outcome vs. dopamine\n(grouped, not stacked -- known dopamine values only)")
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        fig4_path = os.path.join(directory, "figure2_deposit_outcomes.png")
        plt.savefig(fig4_path, dpi=150, bbox_inches="tight")
        plt.close(fig4)
        print(f"\n[OK] Figure 2 -> {fig4_path}")

        write_markdown_table(
            os.path.join(directory, "table1_deposit_outcomes.md"),
            "Table 1. Deposit-release outcome by dopamine level (% of release events).",
            ["Dopamine", "n"] + [o.replace("_", " ") for o in OUTCOMES],
            [[d, len(by_dopamine[d])] +
             [f"{100 * by_dopamine[d].count(o) / len(by_dopamine[d]):.0f}%" for o in OUTCOMES]
             for d in known_dopamines])
        print(f"[OK] Table 1 -> {os.path.join(directory, 'table1_deposit_outcomes.md')}")


if __name__ == "__main__":
    main()
