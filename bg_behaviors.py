"""
Port of can_seek.h, can_pickup.h, wall_seek.h, corner_seek.h, corner_deposit.h.

Each original file had an IDENTICAL copy-pasted MotorActivity() function;
factored here into motor_activity_from_speeds(). Everything else (the
MovementGenerator state machines) is ported 1:1, including the exact
thresholds and step counts, with the original C++ variable names kept in
comments for traceability.

IMPORTANT: khepera_get_proximity(0..5) calls from the original are replaced
by a `left_ir, right_ir` pair passed in from the sensor adapter (see
bg_sensors.py) -- the adapter is responsible for producing numbers on a
comparable scale to the original's raw Khepera IR sums, since the original
thresholds (200, 1025, 1200, 2000, 2100, 3700...) are all calibrated against
that specific hardware's raw proximity range, not against our normalized
[0,1] PyBullet sensor values.
"""
import numpy as np
from bg_core import (
    NUMBER_OF_CHANNELS, MOTOR_VECTOR,
    LEFT_BACTIVITY, LEFT_FACTIVITY, RIGHT_BACTIVITY, RIGHT_FACTIVITY,
    ARM_BACTIVITY, ARM_FACTIVITY, GRIPPER_BACTIVITY, GRIPPER_FACTIVITY,
    FORWARD_SPEED, BEHTIME, THALAMIC_TONIC,
)


def motor_activity_from_speeds(left_speed, right_speed, arm_speed, gripper_speed):
    """The MotorActivity() helper, identical across all 5 original files."""
    activity = np.zeros(MOTOR_VECTOR)

    if left_speed < 0:
        activity[LEFT_BACTIVITY] = -left_speed / FORWARD_SPEED
    elif left_speed > 0:
        activity[LEFT_FACTIVITY] = left_speed / FORWARD_SPEED

    if right_speed < 0:
        activity[RIGHT_BACTIVITY] = -right_speed / FORWARD_SPEED
    elif right_speed > 0:
        activity[RIGHT_FACTIVITY] = right_speed / FORWARD_SPEED

    if arm_speed < 0:
        activity[ARM_BACTIVITY] = -arm_speed
    elif arm_speed > 0:
        activity[ARM_FACTIVITY] = arm_speed

    if gripper_speed < 0:
        activity[GRIPPER_BACTIVITY] = -gripper_speed
    elif gripper_speed > 0:
        activity[GRIPPER_FACTIVITY] = gripper_speed

    return activity


def _clip_speed(v):
    return max(-FORWARD_SPEED, min(FORWARD_SPEED, v))


# DEVIATION FROM THE ORIGINAL (confirmed with the user, not a blind change):
# the ported "avoid" turn was (diff)/30, which goes to ~0 whenever left and
# right IR are nearly balanced -- exactly the glancing angle of a real
# pinball-style wall bounce. With the sensor cone added later (wider
# acceptance per sensor), near-balanced readings got MORE common, so the
# robot's evasive turns became imperceptibly small and it stopped bouncing
# off walls the way it did before the cone. This enforces a minimum turn
# magnitude so a "close but nearly balanced" reading still produces a real,
# visible turn instead of a near-zero one.
MIN_AVOID_TURN = 3.0


def _avoid_turn(left, right, divisor=30.0):
    """Proportional turn away from the stronger side, with a floor so a
    near-balanced reading still produces a real turn (see MIN_AVOID_TURN
    note above)."""
    diff = left - right
    magnitude = max(abs(diff) / divisor, MIN_AVOID_TURN)
    if diff >= 0:
        return magnitude, -magnitude
    else:
        return -magnitude, magnitude


# DEVIATION FROM THE ORIGINAL, confirmed with the user: when nothing is
# detected (total < 200), the original drives dead straight at full speed,
# with zero turning bias, forever, until it hits something. In a square,
# symmetric, mostly-empty room this can lock into a stable periodic orbit
# -- confirmed with real log data: 854 consecutive decision steps (~7
# simulated minutes) tracing a large diamond corner-to-corner, NEVER
# closer than 15cm to the room's centre, while every can sits within 11cm
# of it. A small random turn bias on every "wander" step keeps long
# unobstructed stretches from being perfectly straight, so the trajectory
# gradually curves instead of retracing the same bouncing path -- without
# biasing it toward the centre specifically (that would be a much bigger
# deviation), just breaking the perfect-line degeneracy that causes the
# lock-in.
#
# STILL HAPPENS SOMETIMES, confirmed with the user (real logs, "depends
# on chance"): this bias is fresh i.i.d. Gaussian noise on every single
# wander call, while _avoid_turn() (the wall-bounce) is fully
# deterministic -- a clean mirror reflection of whatever heading arrives.
# So the heading only does a proper random walk if a wander stretch is
# long enough, in steps, for enough independent draws to accumulate a
# real deviation before the next (geometry-preserving) bounce resets
# nothing but simply reflects whatever heading shows up. On a short
# diamond leg, that accumulation sometimes is not enough before the next
# bounce, so it can stay close to periodic for a while. First, minimal
# step tried: just make each draw bigger (doubled, was 1.5) so the same
# few steps of a short leg accumulate more heading change -- before
# reaching for a correlated/random-walk bias or touching _avoid_turn
# itself (bigger deviations, not tried yet).
WANDER_TURN_BIAS_STD = 3.0


def _wander_speeds():
    """FORWARD_SPEED with a small random per-call turn bias (see
    WANDER_TURN_BIAS_STD note above). Same random bias formula used by
    all 3 "nothing detected -> go straight" branches (can_seek, wall_seek,
    corner_seek's >800 mid band uses its own fixed value and is untouched)."""
    bias = np.random.normal(0.0, WANDER_TURN_BIAS_STD)
    return FORWARD_SPEED - bias, FORWARD_SPEED + bias


class ScanCanSeek:
    """can_seek.h :: class scan"""

    def __init__(self):
        self.stm = 0

    def movement_generator(self, thalamus, left_ir, right_ir):
        bhfeedback = 0.0

        if thalamus < THALAMIC_TONIC:
            self.stm = 0
            return motor_activity_from_speeds(0, 0, 0, 0), bhfeedback, None

        if self.stm >= BEHTIME:
            self.stm = 0
            return motor_activity_from_speeds(0, 0, 0, 0), bhfeedback, None

        self.stm += 1
        left, right = left_ir, right_ir
        total = left + right

        if total < 200:
            left_speed, right_speed = _wander_speeds()
        elif total > 2100:
            if left != right:
                left_speed, right_speed = _avoid_turn(left, right)
            else:
                left_speed = right_speed = -FORWARD_SPEED / 2
        elif total > 1025:
            # DEVIATION FROM THE ORIGINAL, conservative and isolated (confirmed
            # with the user): the original sniff band is pure rotation, no
            # forward component at all. With near-tied left/right + our sensor
            # noise, the sign can flip every step, so the robot can oscillate
            # turning in place forever with zero net progress (confirmed with
            # real log data: total stayed ~1830-1870 while left/right kept
            # swapping which was bigger, position frozen for hundreds of
            # steps). This adds a small FIXED forward nudge alongside the
            # turn -- it survives the turn's sign flipping (it's added, not
            # multiplied), so there's always some net advance. Kept small on
            # purpose so it doesn't change the character of "sniff" as
            # primarily a turn, and so it doesn't fight whatever this same
            # behaviour needs to do later in the social-task simulator.
            SNIFF_FORWARD_NUDGE = 1.0   # was 0.5 -- too slow to converge,
                                          # sniffing many cycles before lining up
            if left > right:
                left_speed, right_speed = -1 + SNIFF_FORWARD_NUDGE, 1 + SNIFF_FORWARD_NUDGE
            elif right > left:
                left_speed, right_speed = 1 + SNIFF_FORWARD_NUDGE, -1 + SNIFF_FORWARD_NUDGE
            else:
                left_speed = right_speed = SNIFF_FORWARD_NUDGE
        else:
            left_speed = right_speed = 3

        left_speed, right_speed = _clip_speed(left_speed), _clip_speed(right_speed)

        if left_speed == 0 and right_speed == 0:
            self.stm = BEHTIME   # "can found"

        return motor_activity_from_speeds(left_speed, right_speed, 0, 0), bhfeedback, None


class GripCanPickup:
    """
    can_pickup.h :: class grip -- originally a fixed action pattern of 20
    steps; now 31 (4 back-off/open + 4 idle pause + 8 down + 10 close +
    5 up -- see the phase-length deviation note below).

    The single-hinge arm (khepera_real.urdf) does not have a separate
    fold/unfold joint anymore -- ARM_FACTIVITY/ARM_BACTIVITY drive the one
    real hinge directly (mapped to UP/MIDDLE/DOWN in bg_motor.py's
    motor_plant). hinge_target is kept in the return tuple for interface
    stability but is always None now.

    DEVIATION FROM THE ORIGINAL, confirmed with the user ("functional
    fragility" -- keep it fragile in spirit, but make it actually able to
    finish): confirmed with real log data that the sequence's own first
    phase (back off + open) moves the robot far enough that CAN_DETECTOR
    drops, which lets thalamus decay below THALAMIC_TONIC mid-sequence --
    and the original's `if thalamus < THALAMIC_TONIC: reset` cancels the
    whole grab right there, every time this happens, before it ever closes
    the fingers or lifts the arm. A short grace period (GRACE_STEPS) lets
    a brief dip ride through without aborting, while a longer interruption
    still cancels it -- still fragile, just not self-defeating by design.
    """

    GRACE_STEPS = 15
    COMMIT_STEP = 8  # see note below -- once the arm starts to
                      # lower (end of back-off + pause), the sequence becomes
                      # a genuinely fixed action pattern and is no longer
                      # aborted by thalamus, regardless of what happens.

    def __init__(self):
        self.stm = 0
        self.grace = 0

    def movement_generator(self, thalamus):
        bhfeedback = 0.0
        hinge_target = None

        # DEVIATION FROM THE ORIGINAL, confirmed with the user: GRACE_STEPS
        # (above) was intended for a brief/noisy thalamus drop, but real log
        # data and the new can_pickup_stm confirmed that the drop in
        # CAN_DETECTOR when grabbing the can (already documented below) is not
        # brief -- it lasts for the remainder of the sequence, and GRACE_STEPS
        # is exhausted just as the closing phase begins (stm~16) practically
        # every time. This leaves the outcome to the exact position within
        # the down/close/up cycle at which the abort occurs (sometimes the arm
        # has already re-closed and the can is saved; sometimes it has not and
        # the can is dropped). A genuinely fixed action pattern, once
        # committed (the arm is already lowering, stm>=COMMIT_STEP), is not
        # aborted halfway through even if thalamus drops -- it runs to
        # completion. Before that point (back-off + pause, still reversible
        # and without physical commitment), it can still abort as before.
        if self.stm < self.COMMIT_STEP and thalamus < THALAMIC_TONIC:
            if self.stm > 0 and self.grace < self.GRACE_STEPS:
                self.grace += 1
                # keep running this step as if thalamus were fine -- fall
                # through to the normal stm-based logic below
            else:
                self.stm = 0
                self.grace = 0
                return motor_activity_from_speeds(0, 0, 0, 0), bhfeedback, hinge_target
        else:
            self.grace = 0

        if self.stm >= 31:
            self.stm = 0
            return motor_activity_from_speeds(0, 0, 0, 0), bhfeedback, hinge_target

        # DEVIATION FROM THE ORIGINAL, confirmed with the user: reverted
        # down/close/up back to 8/10/5 (the 15/15/10 version caused a new
        # problem -- continuous strong closing force over 15 steps made the
        # fingers overshoot/bounce off the can back and forth instead of
        # settling, described as "chewing" it). Added a separate 4-step
        # IDLE pause (zero motion) after the back-off motion instead -- the
        # actual back-off distance was already fine according to the user,
        # but the phase felt rushed to watch; this makes it last longer in
        # wall-clock time without moving the robot any farther.
        if self.stm < 4:
            # Speed reduced from the original -3.0: at that speed, over the
            # 5 decision steps of this phase, the robot ended up ~6cm
            # farther from the can than when it started -- more than the
            # arm's own reach (~6-7cm total from the pivot), so it was
            # backing itself out of range before ever trying to grab.
            # Confirmed with real log data (can position never moved,
            # dist_nearest_can climbed from 0.04 to 0.10 during this phase).
            act = motor_activity_from_speeds(-1.0, -1.0, 0.0, -1.0)  # back off, open gripper
            bhfeedback = 1.0
        elif self.stm < 8:
            act = motor_activity_from_speeds(0.0, 0.0, 0.0, -1.0)    # idle pause, gripper stays open
            bhfeedback = 1.0
        elif self.stm < 16:
            act = motor_activity_from_speeds(0.0, 0.0, 1.0, 0.0)     # stop, arm down
            bhfeedback = 1.0
        elif self.stm < 26:
            act = motor_activity_from_speeds(0.0, 0.0, 0.0, 1.0)     # close gripper
            bhfeedback = 1.0
        else:
            act = motor_activity_from_speeds(0.0, 0.0, -1.0, 0.0)    # arm up

        self.stm += 1
        return act, bhfeedback, hinge_target


class SWallSeek:
    """wall_seek.h :: class swall"""

    def __init__(self):
        self.stm = 0

    def movement_generator(self, thalamus, left_ir, right_ir):
        bhfeedback = 0.0

        if thalamus < THALAMIC_TONIC:
            self.stm = 0
            return motor_activity_from_speeds(0, 0, 0, 0), bhfeedback, None

        if self.stm >= BEHTIME:
            self.stm = 0
            return motor_activity_from_speeds(0, 0, 0, 0), bhfeedback, None

        self.stm += 1
        left, right = left_ir, right_ir
        total = left + right

        if total < 200:
            left_speed, right_speed = _wander_speeds()
        elif total < 2100:
            if left != right:
                left_speed, right_speed = _avoid_turn(left, right)
            else:
                left_speed = right_speed = -FORWARD_SPEED / 2
        else:
            left_speed = right_speed = 0

        left_speed, right_speed = _clip_speed(left_speed), _clip_speed(right_speed)

        if left_speed == 0 and right_speed == 0:
            self.stm = BEHTIME   # "wall found"

        return motor_activity_from_speeds(left_speed, right_speed, 0, 0), bhfeedback, None


class SCornerSeek:
    """corner_seek.h :: class scorner"""

    def __init__(self):
        self.stm = 0

    def movement_generator(self, thalamus, left_ir, right_ir):
        bhfeedback = 0.0

        if thalamus < THALAMIC_TONIC:
            self.stm = 0
            return motor_activity_from_speeds(0, 0, 0, 0), bhfeedback, None

        if self.stm >= BEHTIME:
            self.stm = 0
            return motor_activity_from_speeds(0, 0, 0, 0), bhfeedback, None

        self.stm += 1
        left, right = left_ir, right_ir
        total = left + right
        left_speed = right_speed = 0

        if total > 3700:
            left_speed = right_speed = 0
        elif total < 200:        # NEW: nothing detected -> go straight, like
                                  # can_seek/wall_seek already do. The
                                  # original never had this branch -- with
                                  # left==right==0 exactly (no sensor noise
                                  # in simulation) it fell into the turn
                                  # branch below and spun in place forever
                                  # with nothing around, confirmed by testing
                                  # the robot alone with no walls/cans.
            left_speed, right_speed = _wander_speeds()
        else:
            if 1200 < total < 2000:
                left_speed = right_speed = 10
            elif total > 2000:
                if left > right:
                    left_speed, right_speed = 3, -3
                else:
                    left_speed, right_speed = -3, 3
            elif total < 1200:
                if left > right:
                    left_speed, right_speed = -3, 3
                else:
                    left_speed, right_speed = 3, -3

        if left_speed == 0 and right_speed == 0:
            self.stm = BEHTIME   # "corner found"

        return motor_activity_from_speeds(left_speed, right_speed, 0, 0), bhfeedback, None


class ReleaseCornerDeposit:
    """
    corner_deposit.h :: class release -- originally a fixed action pattern
    of 15 steps; now 23 (8 down + 10 release + 5 up). Same single-hinge
    note as GripCanPickup above; hinge_target unused now. Same grace-period
    deviation as GripCanPickup above, same reasoning.

    DEVIATION FROM THE ORIGINAL, confirmed with the user: this one was
    never touched while can_pickup went through several rounds of timing
    fixes (5->8/10/5->28->31 total) -- applying the same reasoning here on
    request, before it is actually confirmed to have the same "rushed"
    problem. Down and release get more time to settle (same as can_pickup's
    down/close did); up stays at 5, same as can_pickup's up phase.
    """

    GRACE_STEPS = 15

    def __init__(self):
        self.stm = 0
        self.grace = 0

    def movement_generator(self, thalamus):
        bhfeedback = 0.0
        hinge_target = None

        if thalamus < THALAMIC_TONIC:
            if self.stm > 0 and self.grace < self.GRACE_STEPS:
                self.grace += 1
            else:
                self.stm = 0
                self.grace = 0
                return motor_activity_from_speeds(0, 0, 0, 0), bhfeedback, hinge_target
        else:
            self.grace = 0

        if self.stm >= 23:
            self.stm = 0
            return motor_activity_from_speeds(0, 0, 0, 0), bhfeedback, hinge_target

        if self.stm < 8:
            act = motor_activity_from_speeds(0.0, 0.0, 0.5, 0.0)     # arm down (to MIDDLE)
        elif self.stm < 18:
            act = motor_activity_from_speeds(0.0, 0.0, 0.0, -1.0)    # open gripper (release)
            bhfeedback = 1.0
        else:
            act = motor_activity_from_speeds(0.0, 0.0, -1.0, 0.0)    # arm up

        self.stm += 1
        return act, bhfeedback, hinge_target