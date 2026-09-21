<p align="center">
  <img src="images/foraging_khepera.jpg" alt="Foraging Khepera I simulated in PyBullet" width="800">
</p>

# Walking and Chewing Gum: A Vertebrate Basal Ganglia Robot under Realistic Contact Physics

A Python and PyBullet port of the basal ganglia model by Prescott et al. (2006), featuring the actual geometry of the Khepera I and its gripper, running the same foraging task as the original study within a modern rigid-body physics engine.

This implementation follows the **embedded action selection architecture (EASA)**: action selection (here, the basal ganglia model) is embedded directly in the sensorimotor loop of a physically simulated robot, rather than evaluated in the abstract.

## Contents

- `bg_core.py`, `bg_sensors.py`, `bg_behaviors.py`, `bg_motor.py` — basal ganglia model core, ported constant-by-constant.
- `easa_standalone.py` — complete simulation: robot + foraging task.
- `easa_arena.py` — environment definition.
- `khepera_real.urdf` — actual robot geometry, extracted from Webots files.
- `easa_config.json` — configuration parameters for a run.
- `easa_plot.py`, `easa_live_plot.py` — visualization of motivations and ethogram (live or post-hoc).
- `run_dopamine_sweep.sh` — systematic dopamine sweep, executed in parallel.
- `aggregate_sweep_results.py` — statistical aggregation of the sweep (Kruskal-Wallis, Fisher, Mann-Whitney, Holm-Bonferroni).
- `analyze_deposit_physics.py` — classification of the physical outcome of each deposit attempt.
- `count_aborted_pickups.py`, `analyze_channel_time.py`, `analyze_channel_switching.py` — selection circuit diagnostics.

## Installation

```bash
pip install -r requirements.txt
```

## Basic usage

```bash
python3 easa_standalone.py
```

For a headless run with specific parameters:

```bash
EASA_HEADLESS=1 EASA_DOPAMINE_D1=0.2 EASA_DOPAMINE_D2=0.2 python3 easa_standalone.py
```

For the full dopamine sweep:

```bash
REPS=20 MAX_PARALLEL_JOBS=8 ./run_dopamine_sweep.sh
python3 aggregate_sweep_results.py
```

## Citation

If you use this code, please cite:

> [pending -- fill in with the paper citation once published on arXiv]

## License

MIT. See [LICENSE](LICENSE).
