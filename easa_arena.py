"""
Recreates arena.wbt in PyBullet at its REAL scale (0.6x0.6m), now that the
robot itself (khepera_real.urdf) is also built at real scale. The earlier
0.75m / inflated-wall-height version was a compromise for the old
oversized (75mm) robot design; none of that is needed anymore.

Wall height back to the real 0.02m too -- that only failed before because
our old sensor mount height (~0.035m, e-puck body midpoint) was ABOVE the
wall top. The real sensor height (0.0145m, from Khepera1.proto) is well
below 0.02m, so the real wall height works correctly now.
"""
import pybullet as p

WORLD_SIZE = 0.6              # real, from arena.wbt
WALL_THICKNESS = 0.01         # real
WALL_HEIGHT = 0.03            # real -- works now, sensor height is 0.0145

WALL_COLOR = [0.9, 0.2, 0.27, 1.0]   # all 4 walls red (user's call from
                                      # earlier -- arena.wbt only colored
                                      # one, kept red-all for visibility)

# Cans, converted from arena.wbt's Webots (x, y_height, z) positions to our
# (x, y) ground-plane positions -- unchanged, these were already real-scale
# Webots coordinates, never tied to our robot's (now-corrected) size:
#   (0.1, 0.02, 0.05)   -> (0.1, 0.05)
#   (-0.05, 0.02, 0.1)  -> (-0.05, 0.1)
#   (-0.1, 0.02, -0.05) -> (-0.1, -0.05)
#   (0.05, 0.02, -0.1)  -> (0.05, -0.1)
CAN_POSITIONS = [
    (0.10, 0.05),
    (-0.05, 0.10),
    (-0.10, -0.05),
    (0.05, -0.10),
]
CAN_RADIUS = 0.01
CAN_HEIGHT = 0.04
CAN_FRICTION = 2.0
CAN_ROLLING_FRICTION = 0.05
CAN_SPINNING_FRICTION = 0.05

# Deposit corner: inset from the real 0.3m half-size.
_half = WORLD_SIZE / 2.0
DEPOSIT_CORNER = (_half - 0.05, _half - 0.05)
DEPOSIT_RADIUS = 0.05

# Recomputed for the real 0.05m sensor range (was 0.12-0.20m before) --
# needs much less clearance from walls now to avoid a false corner trigger
# at spawn. Kept off-centre and near (not on top of) the can cluster, same
# reasoning as before: don't spawn boxed in among all 4 cans.
ROBOT_SPAWN_POS = [0.15, -0.15, 0.005]
ROBOT_SPAWN_YAW_DEG = 135.0


def _make_wall(position, half_extents, color):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    wall_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                 baseVisualShapeIndex=vis, basePosition=position)
    p.changeDynamics(wall_id, -1, lateralFriction=0.5)
    return wall_id


def build_walls():
    half = WORLD_SIZE / 2.0
    wall_ids = []

    for z in (-half, half):
        wall_ids.append(_make_wall(
            [0, z, WALL_HEIGHT / 2],
            [WORLD_SIZE / 2, WALL_THICKNESS / 2, WALL_HEIGHT / 2],
            WALL_COLOR))

    wall_ids.append(_make_wall(
        [-half, 0, WALL_HEIGHT / 2],
        [WALL_THICKNESS / 2, WORLD_SIZE / 2, WALL_HEIGHT / 2],
        WALL_COLOR))
    wall_ids.append(_make_wall(
        [half, 0, WALL_HEIGHT / 2],
        [WALL_THICKNESS / 2, WORLD_SIZE / 2, WALL_HEIGHT / 2],
        WALL_COLOR))

    return wall_ids


def spawn_can(pos_xy):
    x, y = pos_xy
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=CAN_RADIUS, height=CAN_HEIGHT)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=CAN_RADIUS, length=CAN_HEIGHT,
                               rgbaColor=[0.9, 0.1, 0.1, 1])
    can_id = p.createMultiBody(baseMass=0.01, baseCollisionShapeIndex=col,
                                baseVisualShapeIndex=vis,
                                basePosition=[x, y, CAN_HEIGHT / 2])
    p.changeDynamics(can_id, -1, lateralFriction=CAN_FRICTION,
                      rollingFriction=CAN_ROLLING_FRICTION,
                      spinningFriction=CAN_SPINNING_FRICTION,
                      ccdSweptSphereRadius=CAN_RADIUS)
    return can_id


def build_easa_arena():
    """Builds the full arena. Returns a dict with everything the standalone
    script needs to track (can ids + starting positions, deposit info)."""
    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=0.5)

    build_walls()

    can_ids = []
    can_start_positions = []
    for pos in CAN_POSITIONS:
        can_id = spawn_can(pos)
        can_ids.append(can_id)
        can_start_positions.append((pos[0], pos[1], CAN_HEIGHT / 2))

    return {
        "can_ids": can_ids,
        "can_start_positions": can_start_positions,
        "deposit_corner": DEPOSIT_CORNER,
        "deposit_radius": DEPOSIT_RADIUS,
    }
