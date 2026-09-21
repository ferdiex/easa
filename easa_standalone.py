#!/usr/bin/env python3
"""
EASA, complete integration with the REAL geometry (khepera_real.urdf):
sensors at the real Khepera1.proto angles, a single-hinge arm (not the old
hinge+prismatic design), a real-scale arena (0.6x0.6 m), and an optical
barrier in the gripper with the self-hit filter.

This file does not recreate anything from memory -- it was built by reading the
current real state of bg_core.py, bg_motor.py, bg_behaviors.py, bg_sensors.py,
easa_arena.py, khepera_real.urdf, and khepera_real_test.py in this same
session.

Usage:
    python easa_standalone.py [config_path]

Controls:
    P     : pause / resume
    R     : reset (robot, cans, and all internal BG state)
    C     : force the can_pickup sequence to start immediately
            (diagnostic -- the clean equivalent of holding the can with
            the mouse to force detection)
    D     : force the corner_deposit sequence to start immediately
            (same mechanism as C, to test releasing without depending on
            CORNER_DETECTOR firing on its own; also resets hunger to its
            initial value, solely to allow repeated testing without waiting
            for it to increase on its own)
    L     : respawn the cans at their original positions (does not affect
            the robot or fear/hunger/BG state -- to clear fallen or stuck cans
            without losing the run context)
    T     : enable/disable diagnostic ray drawing (sensor readings continue
            to work normally; only drawing is disabled)
    SPACE : fast mode (skips the sleep between physics steps)
    G     : show/hide PyBullet panels (native shortcut)
    Ctrl+C or closing the window: exit

The CSV log uses the same columns as previous versions (compatible with
easa_plot.py and easa_live_plot.py without changes), plus the diagnostic
columns (position, raw detection, and dot_product) that had already been
added.
"""
import sys
import os
import csv
import time
import json

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    p = None
    pybullet_data = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bg_core import (
    BGController, CAN_SEEK, CAN_PICKUP, WALL_SEEK, CORNER_SEEK, CORNER_DEPOSIT,
    CHANNEL_NAMES, NUMBER_OF_CHANNELS, THALAMIC_TONIC, BEHTIME,
    WALL_DETECTOR, GRIPPER_SENSOR, CAN_DETECTOR, CORNER_DETECTOR,
    UP as ARM_UP, MIDDLE as ARM_MIDDLE, DOWN as ARM_DOWN,
    OPEN as GRIP_OPEN, MAXTIME, TIMESTEP,
)
from bg_motor import gate_signals, integrate_activity, motor_plant
from bg_behaviors import (
    ScanCanSeek, GripCanPickup, SWallSeek, SCornerSeek, ReleaseCornerDeposit,
    MOTOR_VECTOR, WANDER_TURN_BIAS_STD,
)
from bg_sensors import MotivationalSystem, left_right_ir, add_sensor_noise, FEAR_DECAY, HUNGER_GROWTH

NUM_SENSORS = 8
SENSOR_ANGLES_DEG = [63.90, 41.38, 15.52, -15.52, -41.38, -63.90, -160.91, 160.91]
SENSOR_ANGLES = np.radians(SENSOR_ANGLES_DEG)
FRONT_SENSOR_IDX = 2
SENSOR_MOUNT_RADIUS = 0.0285
RAY_START_OFFSET = SENSOR_MOUNT_RADIUS + 0.001
SENSOR_HEIGHT = 0.0145
SENSOR_CONE_HALF_WIDTH_DEG = 8.0
SUBRAYS_PER_SENSOR = 5
_SUBRAY_OFFSETS_RAD = np.radians(
    np.linspace(-SENSOR_CONE_HALF_WIDTH_DEG, SENSOR_CONE_HALF_WIDTH_DEG, SUBRAYS_PER_SENSOR))

ARM_SOFT_DOWN_LIMIT = -0.35


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_link_name_to_index(robot_id):
    mapping = {}
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        mapping[info[12].decode("utf-8")] = j
    return mapping


def read_sensor_rays(robot_id, occluding_link_indices, sensor_range, excluded_body_ids=()):
    base_pos, base_ori = p.getBasePositionAndOrientation(robot_id)
    base_mat = p.getMatrixFromQuaternion(base_ori)
    fwd_vec = np.array([base_mat[0], base_mat[3], 0])
    left_vec = np.array([base_mat[1], base_mat[4], 0])
    center = np.array(base_pos) + np.array([0, 0, SENSOR_HEIGHT])

    all_starts, all_ends = [], []
    for a in SENSOR_ANGLES:
        for offset in _SUBRAY_OFFSETS_RAD:
            sub_a = a + offset
            direction = np.cos(sub_a) * fwd_vec + np.sin(sub_a) * left_vec
            all_starts.append(center + direction * RAY_START_OFFSET)
            all_ends.append(center + direction * sensor_range)

    results = p.rayTestBatch(all_starts, all_ends)

    values = np.zeros(NUM_SENSORS)
    starts, hit_points, colors = [], [], []
    for i in range(NUM_SENSORS):
        best_value = 0.0
        best_hit_point = all_ends[i * SUBRAYS_PER_SENSOR + SUBRAYS_PER_SENSOR // 2]
        best_color = [0, 1, 0]
        for k in range(SUBRAYS_PER_SENSOR):
            res = results[i * SUBRAYS_PER_SENSOR + k]
            hit_id, hit_link, hit_frac = res[0], res[1], res[2]
            if hit_id == -1:
                continue
            if hit_id == robot_id and hit_link not in occluding_link_indices:
                continue
            # DEVIATION FROM THE ORIGINAL, confirmed with the user: the
            # self-hit filter above only ever excluded the robot's OWN
            # body -- a can held in the gripper is a separate PyBullet
            # body, so it was never excluded, and a crooked or off-center
            # grip could place it directly inside a sensor cone. Confirmed
            # with real log data: ds1/ds4 remained near their maximum reading
            # for approximately 350 decision steps in a row, regardless of
            # the robot's yaw (a real wall's reading would vary as the robot
            # turns; one that remains fixed through a full rotation is rigidly
            # attached to the robot, meaning that it is holding something) --
            # while the robot was near the center of the arena, nowhere near
            # a real wall. That false "wall" detection caused wall_detected
            # to fire and corner_seek to spin in place pursuing it, because
            # the "wall" rotated with the robot and could never be steered
            # away from. The filter now also excludes any body ID(s) that the
            # gripper is currently touching, passed in as excluded_body_ids
            # from read_gripper_barrier during this same step, in the same way
            # that the robot's own body is already excluded.
            if hit_id in excluded_body_ids:
                continue
            value = 1.0 - hit_frac
            if value > best_value:
                best_value = value
                best_hit_point = res[3]
                best_color = [1, 0, 0]
        values[i] = best_value
        starts.append(all_starts[i * SUBRAYS_PER_SENSOR + SUBRAYS_PER_SENSOR // 2])
        hit_points.append(best_hit_point)
        colors.append(best_color)
    return values, starts, hit_points, colors


def read_gripper_barrier(robot_id, left_finger_idx, right_finger_idx):
    """DEVIATION FROM THE ORIGINAL, confirmed with the user: this used to be a
    single thin ray between the two finger origins. If the grip became
    crooked or off-center (confirmed to occur as a "crooked grip"), the object
    could be genuinely held but remain outside that exact line, causing the
    ray to miss it even though it was actually gripped. It was changed to use
    real physical contact (p.getContactPoints) between either finger and
    anything that is not the robot itself. The exact angle no longer matters;
    the test only asks whether something is actually touching a finger.

    The function also returns the body IDs in contact (see the
    read_sensor_rays exclusion note) so that the distance-sensor rays can
    exclude whatever is actually being held, in the same way that they
    already exclude the robot's own body."""
    contacts_left = p.getContactPoints(bodyA=robot_id, linkIndexA=left_finger_idx)
    contacts_right = p.getContactPoints(bodyA=robot_id, linkIndexA=right_finger_idx)
    held_body_ids = {c[2] for c in contacts_left if c[2] != robot_id} | \
                    {c[2] for c in contacts_right if c[2] != robot_id}
    blocked = bool(held_body_ids)
    left_pos = p.getLinkState(robot_id, left_finger_idx)[0]
    right_pos = p.getLinkState(robot_id, right_finger_idx)[0]
    return blocked, left_pos, right_pos, held_body_ids


class EasaController:
    def __init__(self, dopamine_d1, dopamine_d2, fear_init=1.0, hunger_init=0.1):
        self.bg = BGController(dopamine_d1, dopamine_d2)
        self.motivation = MotivationalSystem(fear_init, hunger_init)
        self._fear_init = fear_init
        self._hunger_init = hunger_init
        self.behaviours = {
            CAN_SEEK: ScanCanSeek(),
            CAN_PICKUP: GripCanPickup(),
            WALL_SEEK: SWallSeek(),
            CORNER_SEEK: SCornerSeek(),
            CORNER_DEPOSIT: ReleaseCornerDeposit(),
        }
        self.bhfeedback = np.zeros(NUMBER_OF_CHANNELS)
        self._deposit_reset_armed = True
        self._deposit_was_holding = False

    def reset(self):
        self.__init__(self.bg.bg.dopamine_d1, self.bg.bg.dopamine_d2,
                      self._fear_init, self._hunger_init)

    def step(self, ray_values, gripper_barrier_hit, arm_is_down, arm_is_up, force_can_pickup=False,
              force_corner_deposit=False):
        sensors = self.motivation.step(ray_values, gripper_barrier_hit, arm_is_down)

        channel_bg_output = self.bg.step(sensors, self.bhfeedback)
        thalamus = self.bg.channel_thalamus_out

        left_ir, right_ir = left_right_ir(ray_values)

        activities = np.zeros((NUMBER_OF_CHANNELS, MOTOR_VECTOR))
        new_bhfeedback = np.zeros(NUMBER_OF_CHANNELS)

        act, bh, _ = self.behaviours[CAN_SEEK].movement_generator(thalamus[CAN_SEEK], left_ir, right_ir)
        activities[CAN_SEEK], new_bhfeedback[CAN_SEEK] = act, bh

        # Manual override (diagnostic tool, confirmed with the user): forces
        # this channel's movement generator to run as if selected, ignoring
        # what the BG actually decided for it during this step. This makes it
        # possible to test whether the grab sequence itself works independently
        # of whether selection triggers it -- the clean equivalent of nudging the
        # can with the mouse to force detection.
        can_pickup_thalamus = THALAMIC_TONIC if force_can_pickup else thalamus[CAN_PICKUP]
        act, bh, _ = self.behaviours[CAN_PICKUP].movement_generator(can_pickup_thalamus)
        activities[CAN_PICKUP], new_bhfeedback[CAN_PICKUP] = act, bh
        if force_can_pickup:
            # Also force the GATE for this channel (fully selected, 0.0),
            # otherwise gate_signals would still weight its activity by the
            # REAL (unforced) BG output. This was found to dilute the command
            # (for example, causing the arm to move to MIDDLE instead of DOWN)
            # during testing.
            channel_bg_output[CAN_PICKUP] = 0.0

        act, bh, _ = self.behaviours[WALL_SEEK].movement_generator(thalamus[WALL_SEEK], left_ir, right_ir)
        activities[WALL_SEEK], new_bhfeedback[WALL_SEEK] = act, bh

        act, bh, _ = self.behaviours[CORNER_SEEK].movement_generator(thalamus[CORNER_SEEK], left_ir, right_ir)
        activities[CORNER_SEEK], new_bhfeedback[CORNER_SEEK] = act, bh

        act, bh, _ = self.behaviours[CORNER_DEPOSIT].movement_generator(
            THALAMIC_TONIC if force_corner_deposit else thalamus[CORNER_DEPOSIT])
        activities[CORNER_DEPOSIT], new_bhfeedback[CORNER_DEPOSIT] = act, bh
        if force_corner_deposit:
            channel_bg_output[CORNER_DEPOSIT] = 0.0

        # DEVIATION FROM THE ORIGINAL, confirmed with the user (fixes the
        # motivational reset asymmetry): sensory.h/MotivationalSystem.step()
        # only resets hunger on the exact step when the gripper-barrier sensor
        # changes from obstructed to clear WHILE corner_now is ALSO true on
        # that same step. Real log data confirmed that this edge transition
        # almost never coincides (gripper_detected remained 1 well beyond the
        # commanded release in every episode examined). In practice, the
        # robot therefore picks objects up, but the deposit almost never
        # "counts"; hunger remains saturated at 1.0, and the robot returns
        # directly to can_seek while still holding the can. Hunger now also
        # resets as soon as corner_deposit actually COMMANDS the release action
        # at least once during an episode (bhfeedback nonzero during stm 8-17),
        # regardless of whether the full 23-step sequence completes cleanly or
        # is cut short by the grace-period abort. "Attempting to release" is
        # sufficient, matching can_pickup, which never required a fully clean
        # close before allowing the robot to move on. This fires once per
        # episode (re-armed only when stm returns to 0), and only if something
        # was actually detected as held when the episode began, so a manual
        # override (force_corner_deposit) with an empty gripper cannot reset
        # hunger for free. The original gripper-edge check in
        # MotivationalSystem.step() remains in place; this is an added safety
        # net, not a replacement.
        deposit_stm = self.behaviours[CORNER_DEPOSIT].stm
        if deposit_stm == 0:
            self._deposit_reset_armed = True
        elif deposit_stm == 1:
            self._deposit_was_holding = bool(sensors[GRIPPER_SENSOR] == 1.0)

        if (new_bhfeedback[CORNER_DEPOSIT] != 0.0 and self._deposit_reset_armed
                and self._deposit_was_holding):
            self.motivation.hunger = float(np.clip(self.motivation.fear / 3.0, 0.0, 1.0))
            self._deposit_reset_armed = False

        self.bhfeedback = new_bhfeedback

        gate_signals(channel_bg_output, activities)
        motor_activity = integrate_activity(activities)
        cmd = motor_plant(motor_activity)

        # DEVIATION FROM THE ORIGINAL, confirmed with the user: only
        # can_pickup and corner_deposit ever affect the arm. If can_pickup
        # aborts mid-sequence (grace period exhausted) while still holding
        # something with the arm down, nothing else in the system ever commands
        # the arm again. Real log data confirmed one episode lasting
        # 242 seconds in which the arm remained pinned at the down limit, the
        # gripper continued holding something, and zero arm commands were
        # issued for the entire interval. A first attempt fixed this inside
        # GripCanPickup itself, but gate_signals zeroed the command because
        # the channel was no longer actually selected, causing its gate to
        # multiply the recovery command by 0. This was tested and confirmed
        # before being sent. The following safety check acts on the FINAL
        # command, after gating, so it cannot be diluted: if something is
        # genuinely held, the arm is down, and NO channel is actively
        # commanding the arm during this step, force it back up.
        # BUG FOUND AND FIXED before this behavior was confirmed: the check
        # below did not distinguish "nobody is managing the arm at all" from
        # "can_pickup is legitimately mid-sequence and deliberately keeping
        # the arm down while closing the gripper" (arm_target is None on
        # purpose during that phase). Because the barrier can detect contact
        # DURING closing (while the arm is still down and no arm command is
        # issued that step), the safety net fired in the middle of the
        # sequence and pulled the arm up before the gripper finished closing.
        # This happened even with the manual override (C), because
        # force_can_pickup only bypasses the BG's own thalamus gate and does
        # not bypass this separate safety check. Added stm==0 as a precondition
        # for both sequences: the safety check triggers only when NEITHER
        # can_pickup NOR corner_deposit is actually in progress.
        no_sequence_running = (self.behaviours[CAN_PICKUP].stm == 0 and
                                self.behaviours[CORNER_DEPOSIT].stm == 0)
        # EXTENDED, at the user's request, confirmed with real logs: this
        # safety check previously only examined arm_is_down (the DOWN limit),
        # but abandoning a sequence can leave the arm at any intermediate
        # position. One real episode confirmed that it became stuck halfway
        # (neither down nor up) while holding a can, and remained there for
        # 680 decision steps without anyone moving it because arm_is_down was
        # false at that position, so the safety check never fired. It now
        # triggers when the arm is "not up" (not arm_is_up) instead of only
        # when it is "down", covering every intermediate position rather than
        # only the exact lower limit.
        if gripper_barrier_hit and not arm_is_up and cmd["arm_target"] is None and no_sequence_running:
            from bg_core import UP as _SAFETY_UP
            cmd["arm_target"] = _SAFETY_UP

        return cmd, sensors, channel_bg_output, thalamus


def main():
    if p is None:
        print("ERROR: pybullet is not installed in this environment.")
        sys.exit(1)

    import easa_arena

    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "easa_config.json")
    cfg = load_config(config_path)

    # NEW (English from here on, per user request): headless batch mode
    # for running many episodes unattended-overrides are supplied through
    # environment variables so a shell script can sweep parameters without
    # editing easa_config.json for each run.
    #   EASA_HEADLESS=1        -> no GUI window, no real-time throttling
    #   EASA_MAX_DECISION_STEPS=<int> -> auto-exit after N decision steps
    #   EASA_DOPAMINE_D1=<float>, EASA_DOPAMINE_D2=<float> -> override cfg
    #   EASA_RUN_TAG=<str>     -> appended to the log/metadata filenames,
    #                              so parallel runs started in the same
    #                              second do not collide
    #   EASA_CONVERGE=1        -> NEW: run BGController's inner
    #                              thalamocortical loop to convergence
    #                              at each decision step (per Prescott,
    #                              Montes Gonzalez et al. 2024,
    #                              Biomimetics 9(3):139, which reports that
    #                              the real Khepera I hardware did this at
    #                              approximately 7 Hz) instead of using the
    #                              fixed MAXTIME/TIMESTEP iteration count.
    #                              See BGController.CONVERGE_MODE in
    #                              bg_core.py for the period-2 limit-cycle
    #                              note. This is meaningfully slower
    #                              (approximately 20-25 times more inner
    #                              iterations) than the default, so budget
    #                              accordingly for a sweep.
    headless = os.environ.get("EASA_HEADLESS", "0") == "1"
    max_decision_steps = os.environ.get("EASA_MAX_DECISION_STEPS")
    max_decision_steps = int(max_decision_steps) if max_decision_steps else None
    if os.environ.get("EASA_DOPAMINE_D1"):
        cfg["bg"]["dopamine_d1"] = float(os.environ["EASA_DOPAMINE_D1"])
    if os.environ.get("EASA_DOPAMINE_D2"):
        cfg["bg"]["dopamine_d2"] = float(os.environ["EASA_DOPAMINE_D2"])
    run_tag = os.environ.get("EASA_RUN_TAG", "")
    if os.environ.get("EASA_CONVERGE", "0") == "1":
        BGController.CONVERGE_MODE = True

    urdf_path = cfg["robot"]["urdf_path"]
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), urdf_path)
    if not os.path.exists(urdf_path):
        print(f"ERROR: URDF not found at {urdf_path}")
        sys.exit(1)

    # NEW, at the user's request: until now, every run always overwrote
    # the same easa_log.csv specified in the configuration, so a new run
    # silently erased the previous one. In addition, the CSV itself did not
    # record the MAXTIME, dopamine values, or other parameters used to create
    # it, so they had to be remembered or guessed (this happened more than
    # once during the session). The configuration filename is used as the
    # base, but each run adds a timestamp before the extension so runs cannot
    # overwrite one another accidentally. The parameters needed to reproduce
    # the run -- not only dopamine and MAXTIME, but everything modified during
    # testing -- are written separately to a .json file with the same base name.
    # They are not written into the CSV, so easa_plot.py and
    # easa_live_plot.py remain compatible (they use csv.DictReader and expect
    # the first line to be the header).
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{run_tag}" if run_tag else ""
    log_path_base, log_path_ext = os.path.splitext(cfg["output"]["log_path"])
    log_path = f"{log_path_base}_{run_timestamp}{tag_suffix}{log_path_ext}"
    log_every = cfg["output"].get("log_every_n_steps", 5)

    p.connect(p.DIRECT if headless else p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, cfg["world"]["gravity"])
    # The user's suspicion was visually confirmed: the fingers crossed their
    # own hard joint limit, which should not be possible. At this scale
    # (millimeters), a minimal numerical solver drift under contact load
    # appears large in proportion. More iterations provide a better solution
    # for the joint-limit and contact constraints simultaneously, at the cost
    # of somewhat higher CPU usage per step.
    p.setPhysicsEngineParameter(numSolverIterations=150)

    arena = easa_arena.build_easa_arena()

    ROBOT_START_POS = easa_arena.ROBOT_SPAWN_POS
    ROBOT_START_ORN = p.getQuaternionFromEuler([0, 0, np.radians(easa_arena.ROBOT_SPAWN_YAW_DEG)])
    robot_id = p.loadURDF(urdf_path, ROBOT_START_POS, ROBOT_START_ORN, useFixedBase=False)

    fr = cfg["robot"]["friction"]
    p.changeDynamics(robot_id, -1, linearDamping=cfg["robot"]["linear_damping"],
                      angularDamping=cfg["robot"]["angular_damping"], lateralFriction=fr["chassis"])

    name_to_link = build_link_name_to_index(robot_id)
    left_finger_idx = name_to_link.get("left_finger")
    right_finger_idx = name_to_link.get("right_finger")
    arm_idx = name_to_link.get("arm_link")
    left_wheel_idx = name_to_link.get("left_wheel")
    right_wheel_idx = name_to_link.get("right_wheel")
    caster_idx = name_to_link.get("caster")
    occluding_link_indices = {name_to_link[n] for n in
                               ("arm_link", "gripper_base", "left_finger", "right_finger")
                               if n in name_to_link}

    if left_wheel_idx is not None:
        p.changeDynamics(robot_id, left_wheel_idx, lateralFriction=fr["wheel"])
    if right_wheel_idx is not None:
        p.changeDynamics(robot_id, right_wheel_idx, lateralFriction=fr["wheel"])
    if caster_idx is not None:
        p.changeDynamics(robot_id, caster_idx, lateralFriction=fr["caster"])
    if left_finger_idx is not None:
        p.changeDynamics(robot_id, left_finger_idx, lateralFriction=fr.get("gripper", 10.0))
    if right_finger_idx is not None:
        p.changeDynamics(robot_id, right_finger_idx, lateralFriction=fr.get("gripper", 10.0))

    arm_limits = p.getJointInfo(robot_id, arm_idx)[8:10] if arm_idx is not None else (0, 0)
    left_finger_limits = p.getJointInfo(robot_id, left_finger_idx)[8:10] if left_finger_idx is not None else (0, 0)
    right_finger_limits = p.getJointInfo(robot_id, right_finger_idx)[8:10] if right_finger_idx is not None else (0, 0)

    if arm_idx is not None:
        p.resetJointState(robot_id, arm_idx, arm_limits[1])

    p.resetDebugVisualizerCamera(cameraDistance=0.35, cameraYaw=30, cameraPitch=-40,
                                  cameraTargetPosition=[0, 0, 0.02])

    controller = EasaController(cfg["bg"]["dopamine_d1"], cfg["bg"]["dopamine_d2"],
                                  cfg["bg"].get("fear_init", 1.0), cfg["bg"].get("hunger_init", 0.1))

    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)

    # See the note above: run metadata is stored in a separate .json file,
    # using the same base name as the log, so the parameters used for the run
    # can be determined later.
    metadata_path = f"{log_path_base}_{run_timestamp}{tag_suffix}.json"
    with open(metadata_path, "w") as meta_file:
        json.dump({
            "run_timestamp": run_timestamp,
            "run_tag": run_tag,
            "log_path": log_path,
            "config_path": config_path,
            "dopamine_d1": cfg["bg"]["dopamine_d1"],
            "dopamine_d2": cfg["bg"]["dopamine_d2"],
            "fear_init": cfg["bg"].get("fear_init", 1.0),
            "hunger_init": cfg["bg"].get("hunger_init", 0.1),
            "MAXTIME": MAXTIME,
            "CONVERGE_MODE": BGController.CONVERGE_MODE,
            "TIMESTEP": TIMESTEP,
            "THALAMIC_TONIC": THALAMIC_TONIC,
            "FEAR_DECAY": FEAR_DECAY,
            "HUNGER_GROWTH": HUNGER_GROWTH,
            "WANDER_TURN_BIAS_STD": WANDER_TURN_BIAS_STD,
            "COMMIT_STEP": GripCanPickup.COMMIT_STEP,
            "WALL_THICKNESS": easa_arena.WALL_THICKNESS,
            "WALL_HEIGHT": easa_arena.WALL_HEIGHT,
            "WORLD_SIZE": easa_arena.WORLD_SIZE,
            "log_every_n_steps": log_every,
            "urdf_path": urdf_path,
        }, meta_file, indent=2)
    print(f"[OK] Run metadata: {metadata_path}")

    log_writer.writerow(["step", "fear", "hunger"] +
                         [f"bg_{n}" for n in CHANNEL_NAMES] +
                         [f"thalamus_{n}" for n in CHANNEL_NAMES] +
                         [f"dotprod_{n}" for n in CHANNEL_NAMES] +
                         # NEW, at the user's request: to diagnose why
                         # simultaneous selection of two or more channels
                         # (the "Huntington-like" behavior) was not visible
                         # even with very high D2. Previously, only the final
                         # result was logged (bg_<channel> = GPi output, the
                         # winner), without exposing the internal D1/D2/GPe
                         # values for each channel separately. These columns
                         # show whether two channels ever reached similar D1
                         # values (a close competition) and how strongly the
                         # D2 differentiation between channels collapses as
                         # dopamine_d2 increases. Without this information,
                         # claims about the architecture would be speculative.
                         [f"d1_{n}" for n in CHANNEL_NAMES] +
                         [f"d2_{n}" for n in CHANNEL_NAMES] +
                         [f"gpe_{n}" for n in CHANNEL_NAMES] +
                         ["robot_x", "robot_y", "robot_yaw_deg",
                          "arm_angle", "arm_is_down", "arm_target_cmd",
                          "gripper_target_cmd", "finger_openness_target",
                          "left_speed_cmd", "right_speed_cmd",
                          "wall_detected", "gripper_detected", "gripper_barrier_raw",
                          "can_detected", "corner_detected",
                          "ds0", "ds1", "ds2", "ds3", "ds4", "ds5", "ds6", "ds7",
                          "left_ir", "right_ir", "front_ray",
                          "can_seek_stm", "can_seek_found", "can_pickup_stm",
                          "dist_nearest_can", "active_channel", "was_reset",
                          "manual_override", "manual_override_deposit"] +
                         [f"can{i}_x" for i in range(len(arena["can_ids"]))] +
                         [f"can{i}_y" for i in range(len(arena["can_ids"]))] +
                         [f"can{i}_z" for i in range(len(arena["can_ids"]))] +
                         [f"can{i}_tilt_deg" for i in range(len(arena["can_ids"]))])

    print(f"[OK] URDF: {urdf_path}")
    print(f"[OK] Arena: {easa_arena.WORLD_SIZE}x{easa_arena.WORLD_SIZE}m real, "
          f"sensor range {cfg['sensors']['range']}m")
    print(f"[OK] Logging to: {log_path} (every {log_every} decision steps)")
    print(f"[OK] Deposit corner at {arena['deposit_corner']}, radius {arena['deposit_radius']}")
    print("[OK] P: pause | R: reset | C: force can_pickup | D: force corner_deposit | "
          "L: respawn cans | T: toggle rays | SPACE: fast mode | G: toggle panels | Ctrl+C: quit")

    arm_position_target = arm_limits[1]
    finger_openness_target = 1.0

    paused = False
    fast_mode = False
    physics_step = 0
    decision_step = 0
    just_reset = False
    manual_override_active = False
    manual_override_deposit_active = False
    show_rays = True
    had_gripper_detected = False
    held_can_ids = set()  # NEW: see the HELD_CAN_RELEASE_DIST note below

    DECISION_EVERY_N_PHYSICS_STEPS = 24
    # NEW, at the user's request ("double confirmation"): real contact
    # (read_gripper_barrier) remains the ONLY mechanism that can mark an
    # object as "held" -- distance alone is never sufficient, because can_seek
    # needs to see the can precisely while approaching it for pickup, when it
    # is closest. However, real logs confirmed that contact flickers from
    # step to step (sometimes present, sometimes absent) even though the can
    # remains at exactly the same distance and has not actually been released.
    # Therefore, once contact confirms a can, it continues to be excluded
    # while it remains within this distance of the robot, and is considered
    # released only after it genuinely moves away (or contact reconfirms it
    # first).
    HELD_CAN_RELEASE_DIST = 0.08

    try:
        while True:
            keys = {} if headless else p.getKeyboardEvents()
            if ord('p') in keys and keys[ord('p')] & p.KEY_WAS_TRIGGERED:
                paused = not paused
                print(f"[{'PAUSED' if paused else 'RESUMED'}]")

            if 32 in keys and keys[32] & p.KEY_WAS_TRIGGERED:
                fast_mode = not fast_mode
                print(f"[fast mode {'ON' if fast_mode else 'OFF'}]")

            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                if not manual_override_active:
                    manual_override_active = True
                    print(f"[override] step={decision_step} forcing can_pickup to start now")
                else:
                    print(f"[override] step={decision_step} already running, ignoring")

            if ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED:
                if not manual_override_deposit_active:
                    manual_override_deposit_active = True
                    # Fixed: this was originally a custom patch to test
                    # corner_deposit without waiting for hunger to rise on its
                    # own, but it reset the ENTIRE BG state (fear, hunger, and
                    # the stm of all five behaviors), rather than only hunger.
                    # It now modifies only hunger, leaving fear and everything
                    # else unchanged.
                    controller.motivation.hunger = controller._hunger_init
                    print(f"[override] step={decision_step} forcing corner_deposit to start now "
                          f"(hunger reset to {controller._hunger_init})")
                else:
                    print(f"[override] step={decision_step} already running, ignoring")

            if ord('t') in keys and keys[ord('t')] & p.KEY_WAS_TRIGGERED:
                show_rays = not show_rays
                print(f"[rays {'ON' if show_rays else 'OFF'}] step={decision_step}")

            if ord('l') in keys and keys[ord('l')] & p.KEY_WAS_TRIGGERED:
                # NEW, at the user's request: respawn only the cans at their
                # original positions and orientations. Neither the robot nor
                # BG state (fear, hunger, or stm) is affected, unlike "R".
                # This is intended to clear cans that have fallen or become
                # stuck on top of the wall without losing the run context.
                for can_id, start_pos in zip(arena["can_ids"], arena["can_start_positions"]):
                    p.resetBasePositionAndOrientation(can_id, start_pos, [0, 0, 0, 1])
                    p.resetBaseVelocity(can_id, [0, 0, 0], [0, 0, 0])
                held_can_ids = set()  # Nothing remains held after respawning
                print(f"[cans] step={decision_step} cans restored to their original positions.")

            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                p.resetBasePositionAndOrientation(robot_id, ROBOT_START_POS, ROBOT_START_ORN)
                p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])
                for can_id, start_pos in zip(arena["can_ids"], arena["can_start_positions"]):
                    p.resetBasePositionAndOrientation(can_id, start_pos, [0, 0, 0, 1])
                    p.resetBaseVelocity(can_id, [0, 0, 0], [0, 0, 0])
                controller.reset()
                arm_position_target = arm_limits[1]
                finger_openness_target = 1.0
                held_can_ids = set()  # Nothing remains held after resetting
                if arm_idx is not None:
                    p.resetJointState(robot_id, arm_idx, arm_limits[1])
                print(f"[reset] step={decision_step} robot, cans, and BG state back to start")
                just_reset = True

            if paused:
                time.sleep(0.05)
                continue

            physics_step += 1

            if physics_step % DECISION_EVERY_N_PHYSICS_STEPS == 0:
                decision_step += 1

                # NEW: headless batch mode automatic exit-close the log and
                # metadata files cleanly and exit instead of running forever,
                # allowing a shell script to sweep many runs unattended.
                if max_decision_steps is not None and decision_step >= max_decision_steps:
                    print(f"[batch] step={decision_step} reached EASA_MAX_DECISION_STEPS, stopping")
                    sys.exit(0)  # the existing `finally` block below handles cleanup

                # NEW: periodic progress output in headless mode, every 10%
                # of the run (or every 500 steps when no maximum is set), so a
                # long unattended sweep's per-run log file
                # (sweep_log_<tag>.txt) shows how far each run has progressed
                # rather than remaining silent until completion.
                if headless:
                    progress_every = max(1, max_decision_steps // 10) if max_decision_steps else 500
                    if decision_step % progress_every == 0:
                        pct = f" ({100 * decision_step / max_decision_steps:.0f}%)" if max_decision_steps else ""
                        print(f"[progress] step={decision_step}{pct}", flush=True)

                gripper_hit = False
                raw_held_ids = set()
                if left_finger_idx is not None and right_finger_idx is not None:
                    gripper_hit, bstart, bend, raw_held_ids = read_gripper_barrier(
                        robot_id, left_finger_idx, right_finger_idx)
                    if show_rays:
                        p.addUserDebugLine(bstart, bend,
                                            lineColorRGB=[1, 0.5, 0] if gripper_hit else [0, 1, 0],
                                            lineWidth=3, lifeTime=0.15)

                if raw_held_ids:
                    # Real contact occurred during this step -- confirm or
                    # reconfirm that the body is being held.
                    held_can_ids = raw_held_ids
                else:
                    # Contact was absent during this particular step -- retain
                    # the exclusion only while each previously marked body
                    # remains genuinely close (see HELD_CAN_RELEASE_DIST).
                    # Once it moves away, consider it released.
                    robot_pos_now = p.getBasePositionAndOrientation(robot_id)[0]
                    still_close = set()
                    for cid in held_can_ids:
                        can_pos = p.getBasePositionAndOrientation(cid)[0]
                        dist = ((can_pos[0] - robot_pos_now[0]) ** 2 +
                                (can_pos[1] - robot_pos_now[1]) ** 2) ** 0.5
                        if dist <= HELD_CAN_RELEASE_DIST:
                            still_close.add(cid)
                    held_can_ids = still_close

                ray_values, starts, hit_points, colors = read_sensor_rays(
                    robot_id, occluding_link_indices, cfg["sensors"]["range"],
                    excluded_body_ids=held_can_ids)
                ray_values = add_sensor_noise(ray_values, cfg["sensors"].get("noise_std", 0.01))

                arm_state = p.getJointState(robot_id, arm_idx) if arm_idx is not None else (0,)
                arm_is_down = arm_idx is not None and arm_state[0] <= ARM_SOFT_DOWN_LIMIT + 0.02
                # NEW, at the user's request: for the abandoned-arm safety
                # mechanism described in the large note in
                # EasaController.step, the controller must know whether the
                # arm is genuinely up, not merely whether it is NOT at the
                # down limit. Real logs confirmed that the arm can become
                # stuck halfway (neither up nor down), which the previous
                # check did not cover.
                arm_is_up = arm_idx is not None and arm_limits[1] is not None and \
                    arm_state[0] >= arm_limits[1] - 0.02

                cmd, sensors, channel_bg_output, thalamus = controller.step(
                    ray_values, gripper_hit, arm_is_down, arm_is_up,
                    force_can_pickup=manual_override_active,
                    force_corner_deposit=manual_override_deposit_active)

                if manual_override_active and controller.behaviours[CAN_PICKUP].stm == 0:
                    manual_override_active = False
                    print(f"[override] step={decision_step} can_pickup sequence finished, back to normal selection")

                if manual_override_deposit_active and controller.behaviours[CORNER_DEPOSIT].stm == 0:
                    manual_override_deposit_active = False
                    print(f"[override] step={decision_step} corner_deposit sequence finished, back to normal selection")

                gripper_now_detected = sensors[GRIPPER_SENSOR] == 1.0
                if gripper_now_detected != had_gripper_detected:
                    running_now = "+".join(
                        CHANNEL_NAMES[i] for i in range(NUMBER_OF_CHANNELS)
                        if thalamus[i] >= THALAMIC_TONIC
                    ) or "none"
                    _, robot_orn_now = p.getBasePositionAndOrientation(robot_id)
                    yaw_now = np.degrees(p.getEulerFromQuaternion(robot_orn_now)[2])
                    _, ang_vel_now = p.getBaseVelocity(robot_id)
                    print(f"[gripper] step={decision_step} "
                          f"{'OBJECT DETECTED between fingers' if gripper_now_detected else 'nothing detected'}"
                          f" | running now: {running_now} | yaw={yaw_now:.1f} deg "
                          f"| turning={abs(ang_vel_now[2]) > 0.3} (ang_vel_z={ang_vel_now[2]:.2f}) "
                          f"| wall_detected={bool(sensors[WALL_DETECTOR] == 1.0)}")
                    had_gripper_detected = gripper_now_detected

                if cmd["arm_target"] is not None:
                    arm_position_target = {ARM_UP: arm_limits[1], ARM_MIDDLE: 0.0,
                                            ARM_DOWN: ARM_SOFT_DOWN_LIMIT}[cmd["arm_target"]]
                if cmd["gripper_target"] is not None:
                    finger_openness_target = 1.0 if cmd["gripper_target"] == GRIP_OPEN else 0.0

                if left_wheel_idx is not None:
                    from bg_core import FORWARD_SPEED as ORIGINAL_FORWARD_SPEED
                    max_wheel_angvel = cfg["robot"].get("max_wheel_angvel", 35.0)
                    left_angvel = (cmd["left_speed"] / ORIGINAL_FORWARD_SPEED) * max_wheel_angvel
                    right_angvel = (cmd["right_speed"] / ORIGINAL_FORWARD_SPEED) * max_wheel_angvel
                    wheel_force = cfg["robot"].get("wheel_motor_force", 14.0)
                    p.setJointMotorControl2(robot_id, left_wheel_idx, p.VELOCITY_CONTROL,
                                             targetVelocity=left_angvel, force=wheel_force)
                    p.setJointMotorControl2(robot_id, right_wheel_idx, p.VELOCITY_CONTROL,
                                             targetVelocity=right_angvel, force=wheel_force)

                if arm_idx is not None:
                    arm_position_target = max(ARM_SOFT_DOWN_LIMIT, arm_position_target)
                    p.setJointMotorControl2(robot_id, arm_idx, p.POSITION_CONTROL,
                                             targetPosition=arm_position_target,
                                             force=cfg["robot"].get("arm_hinge_force", 6.0))
                if left_finger_idx is not None:
                    lt = left_finger_limits[0] + (left_finger_limits[1] - left_finger_limits[0]) * finger_openness_target
                    rt = right_finger_limits[1] + (right_finger_limits[0] - right_finger_limits[1]) * finger_openness_target
                    finger_max_vel = cfg["robot"].get("finger_max_velocity", 0.03)
                    p.setJointMotorControl2(robot_id, left_finger_idx, p.POSITION_CONTROL,
                                             targetPosition=lt, force=cfg["robot"].get("finger_close_force", 8.0),
                                             maxVelocity=finger_max_vel)
                    p.setJointMotorControl2(robot_id, right_finger_idx, p.POSITION_CONTROL,
                                             targetPosition=rt, force=cfg["robot"].get("finger_close_force", 8.0),
                                             maxVelocity=finger_max_vel)

                if show_rays:
                    for i in range(NUM_SENSORS):
                        p.addUserDebugLine(starts[i], hit_points[i], lineColorRGB=colors[i],
                                            lineWidth=3 if i == FRONT_SENSOR_IDX else 1, lifeTime=0.15)

                if decision_step % log_every == 0:
                    robot_pos, robot_orn = p.getBasePositionAndOrientation(robot_id)
                    robot_yaw_deg = np.degrees(p.getEulerFromQuaternion(robot_orn)[2])
                    left_ir_val, right_ir_val = left_right_ir(ray_values)
                    can_seek_beh = controller.behaviours[CAN_SEEK]
                    can_pickup_beh = controller.behaviours[CAN_PICKUP]  # NEW, at the user's request

                    dist_nearest_can = min(
                        ((p.getBasePositionAndOrientation(cid)[0][0] - robot_pos[0]) ** 2 +
                         (p.getBasePositionAndOrientation(cid)[0][1] - robot_pos[1]) ** 2) ** 0.5
                        for cid in arena["can_ids"]
                    ) if arena["can_ids"] else -1.0

                    active_channel = "+".join(
                        CHANNEL_NAMES[i] for i in range(NUMBER_OF_CHANNELS)
                        if thalamus[i] >= THALAMIC_TONIC
                    ) or "none"

                    # NEW, added at the user's request to diagnose cans that
                    # become stranded on top of the wall or bounce back and
                    # fall inside during depositing: z is height -- a can
                    # standing on the floor has z=CAN_HEIGHT/2=0.02; a higher
                    # value indicates that it is mounted on something, such as
                    # the wall -- and tilt is the angle relative to vertical
                    # (0 degrees = upright, 90 degrees = on its side or
                    # fallen). It is calculated from the direction of the can's
                    # local Z axis after rotation by its current quaternion,
                    # so it does not depend on whether the can fell toward
                    # one side or the other (roll versus pitch), unlike reading
                    # roll and pitch separately.
                    can_states = [p.getBasePositionAndOrientation(cid) for cid in arena["can_ids"]]
                    can_positions = [cs[0] for cs in can_states]
                    can_tilts_deg = []
                    for cs in can_states:
                        rot_matrix = p.getMatrixFromQuaternion(cs[1])
                        world_z_of_local_z = max(-1.0, min(1.0, rot_matrix[8]))
                        can_tilts_deg.append(np.degrees(np.arccos(world_z_of_local_z)))

                    log_writer.writerow([decision_step, sensors[4], sensors[5]] +
                                         list(np.round(channel_bg_output, 4)) +
                                         list(np.round(thalamus, 4)) +
                                         list(np.round(controller.bg.bg.last_dot_product, 4)) +
                                         list(np.round(controller.bg.bg.d1_output, 4)) +
                                         list(np.round(controller.bg.bg.d2_output, 4)) +
                                         list(np.round(controller.bg.bg.gpe_output, 4)) +
                                         [round(robot_pos[0], 4), round(robot_pos[1], 4),
                                          round(float(robot_yaw_deg), 2),
                                          round(float(arm_state[0]), 4), int(arm_is_down),
                                          cmd["arm_target"] if cmd["arm_target"] is not None else "",
                                          cmd["gripper_target"] if cmd["gripper_target"] is not None else "",
                                          round(finger_openness_target, 3),
                                          round(float(cmd["left_speed"]), 2), round(float(cmd["right_speed"]), 2),
                                          int(sensors[WALL_DETECTOR] == 1.0),
                                          int(sensors[GRIPPER_SENSOR] == 1.0), int(gripper_hit),
                                          int(sensors[CAN_DETECTOR] == 1.0),
                                          int(sensors[CORNER_DETECTOR] == 1.0)] +
                                         [round(float(v), 3) for v in ray_values] +
                                         [round(float(left_ir_val), 1), round(float(right_ir_val), 1),
                                          round(float(ray_values[FRONT_SENSOR_IDX]), 3),
                                          can_seek_beh.stm, int(can_seek_beh.stm >= BEHTIME),
                                          can_pickup_beh.stm,
                                          round(float(dist_nearest_can), 4), active_channel,
                                          int(just_reset), int(manual_override_active),
                                          int(manual_override_deposit_active)] +
                                         [round(cp[0], 4) for cp in can_positions] +
                                         [round(cp[1], 4) for cp in can_positions] +
                                         [round(cp[2], 4) for cp in can_positions] +
                                         [round(float(t), 2) for t in can_tilts_deg])
                    just_reset = False
                    log_file.flush()

            p.stepSimulation()
            if not fast_mode and not headless:
                time.sleep(1.0 / 240.0)

    except KeyboardInterrupt:
        pass
    finally:
        log_file.close()
        p.disconnect()
        print(f"[OK] Log saved to {log_path}")


if __name__ == "__main__":
    main()