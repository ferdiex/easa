#!/usr/bin/env bash
# run_dopamine_sweep.sh
#
# Runs easa_standalone.py in headless batch mode across a grid of
# dopamine values (D1=D2, symmetric sweep), with several repetitions per
# value, so success/failure rates can be computed with statistics instead
# of anecdotally (matching the methodology of Prescott, Montes Gonzalez
# et al. 2024, Biomimetics 9(3):139, which used 10 trials per dopamine
# level).
#
# Speed notes: PyBullet's physics stepping (C++) is the real bottleneck
# here, not the (already-vectorized, small) numpy math in the BG circuit
# -- numba would not meaningfully speed up p.stepSimulation() calls, so
# it is not used. The two levers that actually matter are (1) headless
# mode with no GUI and no real-time sleep (already ~10-50x faster than
# watching it run), and (2) running independent episodes in parallel
# across CPU cores, since each run is a fully independent process with
# no shared state. This script does (2): it launches up to
# MAX_PARALLEL_JOBS runs at once and waits for a batch to finish before
# starting the next.
#
# Usage:
#   ./run_dopamine_sweep.sh
#   DOPAMINE_VALUES="0.05 0.1 0.2 0.5 1.0" REPS=5 ./run_dopamine_sweep.sh
#   START_REP=16 DOPAMINE_VALUES="0.05" REPS=25 ./run_dopamine_sweep.sh
#     (adds 25 more reps to just D=0.05, numbered rep16..rep40 instead of
#      restarting at rep1 -- purely for readable filenames; nothing ever
#      overwrites, since every filename already carries the full launch
#      timestamp before the tag)
#   CONVERGE=1 DOPAMINE_VALUES="0.2 0.5" REPS=3 ./run_dopamine_sweep.sh
#     (runs the BG circuit to convergence each decision step instead of
#      the fixed MAXTIME/TIMESTEP count -- see BGController.CONVERGE_MODE
#      in bg_core.py. Meaningfully slower per decision step; start with a
#      small REPS/DOPAMINE_VALUES pilot like this to measure the real
#      wall-clock cost on your machine before committing to a full sweep)
#
# Each run writes its own timestamped easa_log_<timestamp>_<tag>.csv and
# matching .json metadata file (dopamine, MAXTIME, CONVERGE_MODE, all
# tuned constants) next to this script, per the existing per-run file
# naming in easa_standalone.py. Nothing gets overwritten between runs --
# not even across separate invocations of this script re-using the same
# dopamine value and rep numbers, since the timestamp differs.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Dopamine values to sweep (D1=D2=this value, symmetric).
DOPAMINE_VALUES="${DOPAMINE_VALUES:-0.05 0.1 0.2 0.5 1.0}"
REPS="${REPS:-5}"                          # repetitions per dopamine value
START_REP="${START_REP:-1}"                # first rep number (see usage above)
MAX_DECISION_STEPS="${MAX_DECISION_STEPS:-6000}"   # ~10 minutes of sim time at 10Hz
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
CONVERGE="${CONVERGE:-0}"                  # 1 -> EASA_CONVERGE=1 passed to every run

echo "[sweep] dopamine values: $DOPAMINE_VALUES"
echo "[sweep] repetitions per value: $REPS"
echo "[sweep] max decision steps per run: $MAX_DECISION_STEPS"
echo "[sweep] parallel jobs: $MAX_PARALLEL_JOBS"
echo "[sweep] converge mode: $CONVERGE"

# Total number of runs, computed up front so progress prints can show
# "X of N done" instead of just "waiting...".
total_runs=0
for dopamine in $DOPAMINE_VALUES; do
    for rep in $(seq "$START_REP" "$((START_REP + REPS - 1))"); do
        total_runs=$((total_runs + 1))
    done
done
echo "[sweep] total runs planned: $total_runs"

# NEW: waits for the currently-running background jobs, printing how many
# are still active every 10 seconds, so a long sweep isn't silent.
wait_with_progress() {
    while true; do
        running=$(jobs -rp | wc -l | tr -d ' ')
        if [[ "$running" -eq 0 ]]; then
            break
        fi
        echo "[sweep] $running job(s) still running in this batch..."
        sleep 10
    done
    wait
}

run_count=0
for dopamine in $DOPAMINE_VALUES; do
    for rep in $(seq "$START_REP" "$((START_REP + REPS - 1))"); do
        tag="D${dopamine}_rep${rep}"
        if [[ "$CONVERGE" == "1" ]]; then
            tag="${tag}_converge"
        fi
        echo "[sweep] launching run_tag=$tag"
        EASA_HEADLESS=1 \
        EASA_MAX_DECISION_STEPS="$MAX_DECISION_STEPS" \
        EASA_DOPAMINE_D1="$dopamine" \
        EASA_DOPAMINE_D2="$dopamine" \
        EASA_RUN_TAG="$tag" \
        EASA_CONVERGE="$CONVERGE" \
        python3 easa_standalone.py > "sweep_log_${tag}.txt" 2>&1 &

        run_count=$((run_count + 1))
        echo "[sweep] launched $run_count of $total_runs"
        if (( run_count % MAX_PARALLEL_JOBS == 0 )); then
            wait_with_progress   # let this batch finish before starting the next one
        fi
    done
done
wait_with_progress   # catch any remaining jobs from the last, possibly partial, batch

echo "[sweep] done: $run_count runs launched. CSV/JSON pairs are in this directory."
echo "[sweep] next: python3 aggregate_sweep_results.py"
