"""
Port of motor.h. gate_signals() and integrate_activity() are 1:1 with the
original. motor_plant() is NOT a 1:1 port -- the original directly called
Khepera hardware functions (khepera_set_speed, gripper_set_arm,
gripper_set_grip); here it just returns a dict of commands, which the
standalone script then applies to the PyBullet joints.
"""
import numpy as np
from bg_core import (
    NUMBER_OF_CHANNELS, MOTOR_VECTOR, NUMBER_OF_MOTORS,
    LEFT_BACTIVITY, LEFT_FACTIVITY, RIGHT_BACTIVITY, RIGHT_FACTIVITY,
    ARM_BACTIVITY, ARM_FACTIVITY, GRIPPER_BACTIVITY, GRIPPER_FACTIVITY,
    LEFT_MOTOR, RIGHT_MOTOR, ARM_MOTOR, GRIPPER_MOTOR,
    UP, MIDDLE, DOWN, OPEN, CLOSED,
    FORWARD_SPEED, SWITCHING_CONSTANT, SIGNAL_M, SIGNAL_T, ACTIVITY_M, ACTIVITY_T,
    LeakyNeuron,
)


def piecewise_linear(a, m, t):
    """motor.h has its own free-function copy of the same squashing function."""
    return LeakyNeuron.piecewise_linear(a, m, t)


def gate_signals(bg_output, activities):
    """
    bg_output: array (NUMBER_OF_CHANNELS,)
    activities: array (NUMBER_OF_CHANNELS, MOTOR_VECTOR), MODIFIED IN PLACE
                (matches the original's pass-by-reference semantics)
    """
    for i in range(NUMBER_OF_CHANNELS):
        for j in range(MOTOR_VECTOR):
            activities[i, j] = piecewise_linear(
                (1 - SWITCHING_CONSTANT * bg_output[i]) * activities[i, j],
                SIGNAL_M, SIGNAL_T)
    return activities


def integrate_activity(activities):
    """activities: (NUMBER_OF_CHANNELS, MOTOR_VECTOR) -> motor_activity (MOTOR_VECTOR,)"""
    motor_activity = activities.sum(axis=0)
    for i in range(MOTOR_VECTOR):
        motor_activity[i] = piecewise_linear(motor_activity[i], ACTIVITY_M, ACTIVITY_T)
    return motor_activity


def motor_plant(motor_activity):
    """
    NOT a 1:1 port (no hardware here). Returns a dict of high-level commands:
      - 'left_speed', 'right_speed': same formula/units as the original
        (-BACTIVITY + FACTIVITY) * FORWARD_SPEED, i.e. still in the original's
        arbitrary Khepera speed units -- the standalone script's adapter is
        responsible for converting these into PyBullet wheel angular
        velocities, not this function.
      - 'arm_target': one of UP / MIDDLE / DOWN / None (None = no change,
        matches the original's "if both activities are 0, don't touch the
        arm" behaviour)
      - 'gripper_target': OPEN / CLOSED / None (None = no change)
    """
    left_speed = (-motor_activity[LEFT_BACTIVITY] + motor_activity[LEFT_FACTIVITY]) * FORWARD_SPEED
    right_speed = (-motor_activity[RIGHT_BACTIVITY] + motor_activity[RIGHT_FACTIVITY]) * FORWARD_SPEED

    arm_target = None
    if not (motor_activity[ARM_BACTIVITY] == 0.0 and motor_activity[ARM_FACTIVITY] == 0.0):
        if motor_activity[ARM_FACTIVITY] > 0.5:
            arm_target = DOWN
        elif motor_activity[ARM_FACTIVITY] > 0.0:
            arm_target = MIDDLE
        elif motor_activity[ARM_BACTIVITY] > 0.0:
            arm_target = UP

    gripper_target = None
    if motor_activity[GRIPPER_BACTIVITY] > 0.0:
        gripper_target = OPEN
    elif motor_activity[GRIPPER_FACTIVITY] > 0.0:
        gripper_target = CLOSED

    return {
        "left_speed": left_speed,
        "right_speed": right_speed,
        "arm_target": arm_target,
        "gripper_target": gripper_target,
    }
