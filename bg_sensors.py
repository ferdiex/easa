"""
Sensor adapter: PyBullet raycasts -> BG's 6-element binary sensor vector,
following sensory.h's thresholds and logic AS ORIGINALLY WRITTEN, no
rescaling.

This version assumes the REAL Khepera sensor layout (khepera_real.urdf /
khepera_real_test.py): 8 sensors at the real angles extracted from
Khepera1.proto (idx 0,1,2 = left-front at +63.9/+41.4/+15.5 deg, idx 3,4,5
= right-front at -15.5/-41.4/-63.9 deg, idx 6,7 = rear). This is why we can
go back to the original C++ thresholds unchanged: idx[0]+idx[1]+idx[2] and
idx[3]+idx[4]+idx[5] are genuinely 3 distinct real sensors per side (no
overlap), exactly matching the original hardware's own left/right grouping
-- the earlier 2/3 rescaling was only needed because our old 8-ring (evenly
spaced 45 deg apart) didn't have a clean 3+3 split without reusing a
sensor. That workaround is gone now that the geometry is real.

RAW SCALE: normalized ray readings (0=nothing, 1=touching) are scaled by
MAX_IR_PER_SENSOR=1023 to land in the original C++ thresholds' range (a
10-bit ADC, matching the real Khepera1.proto SFH900 sensors' lookupTable
convention: 0 at max range, 1023 at contact).

CAN_DETECTOR: original required its two most-central sensors high AND its
next-most-central pair low. With the real layout this maps exactly onto
idx[2],idx[3] (+-15.5 deg, innermost) high, idx[1],idx[4] (+-41.4 deg,
next pair out) low -- the original's own indices, unchanged.

GRIPPER_SENSOR (optical barrier): computed OUTSIDE this module (needs a
PyBullet raytest between the actual finger links, done in the standalone
script) and passed in as a bool. Must exclude self-hits against the
robot's own body -- found and fixed as a real bug earlier this session
(without the filter, it reads "blocked" constantly, not just when holding
something).

The original's `if (sensors[FEAR] == 0.01) {...}` exact-float-equality
reset was NOT ported -- looks like an incomplete/legacy hook, not
faithfully reproducible in a noisy simulation anyway.
"""
import numpy as np
from bg_core import WALL_DETECTOR, GRIPPER_SENSOR, CAN_DETECTOR, CORNER_DETECTOR, FEAR, HUNGER

MAX_IR_PER_SENSOR = 1023.0

# Real angles, for reference/documentation (the standalone script owns the
# actual raycasting and uses these same angles).
SENSOR_ANGLES_DEG = [63.90, 41.38, 15.52, -15.52, -41.38, -63.90, -160.91, 160.91]

# DEVIATION FROM THE ORIGINAL, added deliberately (confirmed with the user):
# our raycasting is fully deterministic -- a real IR sensor never reads an
# exact zero or a perfect left==right tie. Without noise, the system can
# settle into degenerate fixed points and get stuck there forever (seen
# concretely: thalamus oscillating EXACTLY at the 1.0 selection threshold,
# and corner_seek spinning in place forever with left==right==0.0 exactly).
DEFAULT_SENSOR_NOISE_STD = 0.01   # in the normalized 0..1 ray scale


def add_sensor_noise(ray_values, noise_std=DEFAULT_SENSOR_NOISE_STD, rng=None):
    """Adds small Gaussian noise to the normalized ray readings, clipped
    back to [0, 1]. Call ONCE per decision step, right after reading the
    raw rays -- everything downstream uses the same noisy values."""
    if rng is None:
        rng = np.random
    noisy = np.asarray(ray_values, dtype=float) + rng.normal(0.0, noise_std, size=len(ray_values))
    return np.clip(noisy, 0.0, 1.0)


# Original C++ thresholds, UNCHANGED (no rescaling needed anymore -- see
# module docstring).
WALL_SUM_THRESHOLD = 2100
CORNER_SUM_THRESHOLD = 3700
# DEVIATION FROM THE ORIGINAL, confirmed with the user: 1000 (out of 1023,
# ~98%) required near-perfect contact to trigger CAN_DETECTOR. Combined
# with the narrow geometric window for both inner sensors to hit a 1cm-
# radius can at once (~9mm of usable approach distance, calculated
# earlier), the detector almost never fired on its own even when well
# aligned (confirmed: user had to manually hold the can with the mouse to
# get can_pickup to trigger at all). Loosened to 700 (~68%).
CAN_INNER_THRESHOLD = 700
CAN_OUTER_THRESHOLD = 10

# DEVIATION FROM THE ORIGINAL, confirmed with the user (an initial
# "low-cost" experiment before changing BG frequency): real log data
# confirmed that FEAR never undergoes a reset event and only decays -- in an
# episode of approximately 1600 decision steps, it fell from 0.995 to 0.21
# (a factor of approximately 4.7x). wall_seek and corner_seek weight FEAR at
# 0.9/0.8 (SENSORY_WEIGHTS in bg_core.py), compared with only -0.3 for the
# gripper signal in can_seek, so after a certain point can_seek wins the
# competition even though the robot is already holding something -- this
# matches exactly the behavior seen in the log: "it picks up the second can
# but can no longer find the path back to the wall/corner." Relaxed 0.999
# -> 0.9996 (with this constant, the same 1600 steps would leave FEAR at
# approximately 0.5 instead of approximately 0.21) as the first test value,
# WITHOUT changing DECISION_EVERY_N_PHYSICS_STEPS or dopamine -- this is an
# adjustment independent of the leaky integrator (leaky.h/LeakyNeuron), which
# has its own fixed TIMESTEP=0.2 in bg_core.py and its own stability condition
# (K*TIMESTEP<2); this change does not affect that condition and should not
# produce anything resembling the "Huntingtonian" instability caused by
# changing THAT TIMESTEP by mistake. HUNGER_GROWTH is left unchanged for now
# to isolate the variable -- if necessary, it can be changed later,
# separately.
FEAR_DECAY = 0.9999
HUNGER_GROWTH = 1.005


def left_right_ir(ray_values):
    """ray_values: array of 8 normalized sensor readings (real Khepera
    angles, idx 0,1,2 = left-front, idx 3,4,5 = right-front, idx 6,7 = rear).
    Returns (left_ir, right_ir) on the original's raw 0..3069-per-side
    scale -- 3 distinct real sensors per side, no sharing."""
    raw = np.asarray(ray_values) * MAX_IR_PER_SENSOR
    left_ir = raw[0] + raw[1] + raw[2]
    right_ir = raw[3] + raw[4] + raw[5]
    return left_ir, right_ir


def detect_wall(ray_values):
    left_ir, right_ir = left_right_ir(ray_values)
    return (left_ir + right_ir) > WALL_SUM_THRESHOLD


def detect_corner(ray_values):
    left_ir, right_ir = left_right_ir(ray_values)
    return (left_ir + right_ir) > CORNER_SUM_THRESHOLD


def detect_can(ray_values):
    """Original indices: inner pair (2,3) high, outer-next pair (1,4) low --
    a narrow object dead ahead with nothing beside it."""
    raw = np.asarray(ray_values) * MAX_IR_PER_SENSOR
    return (raw[2] > CAN_INNER_THRESHOLD and raw[3] > CAN_INNER_THRESHOLD and
            raw[1] < CAN_OUTER_THRESHOLD and raw[4] < CAN_OUTER_THRESHOLD)


class MotivationalSystem:
    """Ports sensory.h's `sens` class: turns raw perception into the BG's
    6-element sensor vector, and evolves FEAR/HUNGER over time."""

    def __init__(self, fear_init=1.0, hunger_init=0.1):
        self.fear = fear_init
        self.hunger = hunger_init
        self._prev_gripper_obstructed = -1.0  # matches sens::gripper, init -1

    def step(self, ray_values, gripper_barrier_hit, arm_is_down):
        """
        ray_values: 8 normalized PyBullet sensor readings this step (real angles).
        gripper_barrier_hit: bool, did the finger-tip barrier ray hit something
                              (already self-hit-filtered by the caller).
        arm_is_down: bool, arm position. Kept in the signature for call-site
                     compatibility, but no longer used inside this function
                     (see the GRIPPER_SENSOR deviation note below).

        Returns sensors (np.array shape (6,)), matching bg_controller.c's
        `sensors[]` after RecognitionMotivationalSystems().
        """
        sensors = np.full(6, -1.0)

        if detect_wall(ray_values):
            sensors[WALL_DETECTOR] = 1.0

        # DEVIATION FROM THE ORIGINAL, confirmed with the user: the original
        # C++ condition was `presence && arm != DOWN` -- on real hardware,
        # "presence" stays true continuously once something is between the
        # fingers, regardless of arm angle, so filtering by "not down" just
        # excluded the brief grabbing instant. In our simulation the
        # optical-barrier ray only ever registered a hit WHILE arm_is_down
        # (confirmed with real log data: gripper_barrier_raw was 1 five
        # times in 555 rows, and arm_is_down was ALSO 1 every single one of
        # those times, never once the opposite) -- so this filter was
        # silencing the ONLY moments the signal ever fired, making
        # GRIPPER_SENSOR permanently 0 regardless of whether anything was
        # actually held. Removed; GRIPPER_SENSOR now mirrors the barrier
        # directly.
        gripper_obstructed = bool(gripper_barrier_hit)
        if gripper_obstructed:
            sensors[GRIPPER_SENSOR] = 1.0

        if detect_can(ray_values):
            sensors[CAN_DETECTOR] = 1.0

        corner_now = detect_corner(ray_values)
        if corner_now:
            sensors[CORNER_DETECTOR] = 1.0

        # motivations drift on their own (sensory.h)
        self.fear *= FEAR_DECAY
        self.hunger *= HUNGER_GROWTH

        # "just released something right at a corner" -> satiated
        if (self._prev_gripper_obstructed == 1.0 and
                sensors[GRIPPER_SENSOR] == -1.0 and corner_now):
            self.hunger = self.fear / 3.0

        self.hunger = float(np.clip(self.hunger, 0.0, 1.0))

        sensors[FEAR] = self.fear
        sensors[HUNGER] = self.hunger

        self._prev_gripper_obstructed = sensors[GRIPPER_SENSOR]
        return sensors