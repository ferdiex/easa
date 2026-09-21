"""
Pure Python port of the mathematical core of the basal ganglia controller
(Prescott, Gonzalez, Gurney, Humphries & Redgrave, Neural Networks 2006).

Ported 1:1 from: leaky.h, channel.h, nuclei.h, basal_ganglia.h,
bg_params.h, and the constant definitions in bg_controller.h.

NOT ported here (next phase): sensory.h, motor.h, the 5 movement-generator
modules (can_seek.h, can_pickup.h, wall_seek.h, corner_seek.h,
corner_deposit.h), and the hardware I/O in bg_controller.c's main() loop
(khepera_live(), gripper_enable_presence(), the GUI calls, etc.). Those
need an adapter against the PyBullet sim, not a straight translation.

Every constant below is named after -- and equal to -- its #define in the
original C++, with a comment pointing at the source file, so this can be
diffed against the original headers.
"""
import numpy as np

# ============================================================
# bg_controller.h
# ============================================================
NUMBER_OF_CHANNELS = 5
NUMBER_OF_SENSORS = 6
MOTOR_VECTOR = 8
NUMBER_OF_MOTORS = 4

CAN_SEEK, CAN_PICKUP, WALL_SEEK, CORNER_SEEK, CORNER_DEPOSIT = range(5)
CHANNEL_NAMES = ["can_seek", "can_pickup", "wall_seek", "corner_seek", "corner_deposit"]

WALL_DETECTOR, GRIPPER_SENSOR, CAN_DETECTOR, CORNER_DETECTOR, FEAR, HUNGER = range(6)
SENSOR_NAMES = ["wall_detector", "gripper_sensor", "can_detector", "corner_detector", "fear", "hunger"]

K = 5.0            # rate for the leaky integrators
TIMESTEP = 0.2     # timestep for the leaky integrators 
MAXTIME = 0.2      # time for the basal ganglia (inner loop bound) -- restored to 0.2 (the default): confirmed that MAXTIME=1.2 broke the behavior even at D=0.2, not only the dopamine tests. Increase it deliberately only for targeted frequency/dopamine tests; do not leave it enabled for normal runs.

CORTEX_M, CORTEX_T = 1.0, 0.0
THALAMUS_M, THALAMUS_T = 0.62, -0.8
TRN_M, TRN_T = 0.5, 0.0

W_CTXTH = 1.0
W_BGTH = -1.0
W_THCTX = 1.0
W_CTXTRN = 1.0
W_THTRN = 1.0
W_TRNTH = 0.13

# Motor / behaviour side (not needed for the isolated core sanity check,
# needed now for the full loop + the 5 movement-generator modules)
LEFT_BACTIVITY, LEFT_FACTIVITY, RIGHT_BACTIVITY, RIGHT_FACTIVITY, \
    ARM_BACTIVITY, ARM_FACTIVITY, GRIPPER_BACTIVITY, GRIPPER_FACTIVITY = range(8)

LEFT_MOTOR, RIGHT_MOTOR, ARM_MOTOR, GRIPPER_MOTOR = range(4)

UP, MIDDLE, DOWN = 152, 227, 255   # kept as original raw values, for traceability
OPEN, CLOSED = 0, 1

FORWARD_SPEED = 20
BEHTIME = 200
SWITCHING_CONSTANT = 2.5
SIGNAL_M, SIGNAL_T = 1.0, 0.0
ACTIVITY_M, ACTIVITY_T = 1.0, 0.0
THALAMIC_TONIC = 1.0

# ============================================================
# bg_params.h
# ============================================================
D1_DOPAMINE_DEFAULT = 0.2   # #define D1 0.2
D2_DOPAMINE_DEFAULT = 0.2   # #define D2 0.2

STN_M, STN_T = 0.35, -0.25
W_CTX = 1.0
W_BSTN = 0.1
W_GPE_STN = -1.0

MODULATION = 1.0
W_EXT = 1.0
W_BSTR = 0.1

D1_M, D1_T = 0.35, 0.2
W_HSD1 = 1.0

D2_M, D2_T = 0.35, 0.2
W_HSD2 = 1.0

W_STR = -1.0
W_STN_GPE = 0.8
W_STN_GPI = 0.8

GPE_M, GPE_T = 1.0, -0.2
W_GPE_GPI = -0.4

GPI_M, GPI_T = 1.0, -0.2

# basal_ganglia.h :: bg::Output -- hardcoded per-channel weight matrices
CORTEX_WEIGHTS = np.array([0.0, 0.5, 0.0, 0.9, 0.5])
SENSORY_WEIGHTS = np.array([
    [0.0, -0.3, -0.3, 0.0, 0.0, 0.9],   # can_seek
    [0.0, -0.3, 0.3, 0.0, 0.0, 0.9],    # can_pickup
    [-0.3, 0.3, 0.0, 0.0, 0.9, 0.0],    # wall_seek
    [0.3, 0.3, 0.0, 0.0, 0.8, 0.0],     # corner_seek
    [0.0, 0.75, 0.0, 0.75, 0.0, 0.5],   # corner_deposit
])


# ============================================================
# leaky.h
# ============================================================
class LeakyNeuron:
    """Leaky integrator + piecewise-linear squashing (leaky.h)."""

    def __init__(self, init=0.0):
        self.a = init

    def integration(self, u, k, timestep):
        self.a = self.a + (-k * (self.a - u)) * timestep
        return self.a

    @staticmethod
    def piecewise_linear(a, m, t):
        if a < t:
            return 0.0
        elif t <= a <= 1.0 / m + t:
            return m * (a - t)
        else:
            return 1.0


# ============================================================
# channel.h
# ============================================================
class Channel:
    """Cortex / TRN / thalamus loop for one behaviour channel (channel.h)."""

    def __init__(self):
        self.cortex = LeakyNeuron(0.0)
        self.trn = LeakyNeuron(0.0)
        self.thalamus = LeakyNeuron(0.0)

    def i_salience_cortex(self, th):
        u = self.cortex.integration(th * W_THCTX, K, TIMESTEP)
        return self.cortex.piecewise_linear(u, CORTEX_M, CORTEX_T)

    def trn_output(self, ctx, th):
        u = self.trn.integration(ctx * W_CTXTRN + th * W_THTRN, K, TIMESTEP)
        return self.trn.piecewise_linear(u, TRN_M, TRN_T)

    def thalamus_feedback(self, ctx, bg, trn):
        u = self.thalamus.integration(ctx * W_CTXTH + bg * W_BGTH + trn, K, TIMESTEP)
        return self.thalamus.piecewise_linear(u, THALAMUS_M, THALAMUS_T)


# ============================================================
# nuclei.h
# ============================================================
class STNNeuron:
    def __init__(self):
        self.leaky = LeakyNeuron(0.0)

    def output(self, cortex, wcortex, dotproduct, gpe, bhfeedback):
        u = self.leaky.integration(
            (cortex * (W_CTX + wcortex) + dotproduct + bhfeedback * W_BSTN) + (gpe * W_GPE_STN),
            K, TIMESTEP)
        return self.leaky.piecewise_linear(u, STN_M, STN_T)


class D1Neuron:
    def __init__(self):
        self.leaky = LeakyNeuron(0.0)

    def output(self, cortex, wcortex, dotproduct, bhfeedback, recd1, dopamine_d1):
        u = self.leaky.integration(
            (MODULATION + dopamine_d1) *
            (cortex * (W_EXT + wcortex) + dotproduct + bhfeedback * W_BSTR) + recd1,
            K, TIMESTEP)
        return self.leaky.piecewise_linear(u, D1_M, D1_T)


class D2Neuron:
    def __init__(self):
        self.leaky = LeakyNeuron(0.0)

    def output(self, cortex, wcortex, dotproduct, bhfeedback, recd2, dopamine_d2):
        u = self.leaky.integration(
            (MODULATION - dopamine_d2) *
            (cortex * (W_EXT + wcortex) + dotproduct + bhfeedback * W_BSTR) + recd2,
            K, TIMESTEP)
        return self.leaky.piecewise_linear(u, D2_M, D2_T)


class GPeNeuron:
    def __init__(self):
        self.leaky = LeakyNeuron(0.0)

    def output(self, d2, stn):
        u = self.leaky.integration(d2 * W_STR + stn * W_STN_GPE, K, TIMESTEP)
        return self.leaky.piecewise_linear(u, GPE_M, GPE_T)


class GPiNeuron:
    def __init__(self):
        self.leaky = LeakyNeuron(0.0)

    def output(self, d1, stn, gpe):
        u = self.leaky.integration(d1 * W_STR + stn * W_STN_GPI + gpe * W_GPE_GPI, K, TIMESTEP)
        return self.leaky.piecewise_linear(u, GPI_M, GPI_T)


# ============================================================
# basal_ganglia.h
# ============================================================
class BasalGanglia:
    """Wires the 5 nuclei classes across all channels (basal_ganglia.h::bg).

    dopamine_d1 / dopamine_d2 replace the hardcoded #define D1/D2 0.2 --
    exposed as parameters so they can be swept (that's the whole point of
    this sanity check: this is literally the mechanism the original paper
    used to move the same robot between normal / Parkinsonian /
    Huntingtonian-like regimes).
    """

    def __init__(self, dopamine_d1=D1_DOPAMINE_DEFAULT, dopamine_d2=D2_DOPAMINE_DEFAULT):
        n = NUMBER_OF_CHANNELS
        self.stn = [STNNeuron() for _ in range(n)]
        self.d1 = [D1Neuron() for _ in range(n)]
        self.d2 = [D2Neuron() for _ in range(n)]
        self.gpe = [GPeNeuron() for _ in range(n)]
        self.gpi = [GPiNeuron() for _ in range(n)]
        self.stn_output = np.zeros(n)
        self.d1_output = np.zeros(n)
        self.d2_output = np.zeros(n)
        self.gpe_output = np.zeros(n)
        self.dopamine_d1 = dopamine_d1
        self.dopamine_d2 = dopamine_d2
        self.last_dot_product = np.zeros(n)

    def output(self, cortex, sensors, bhfeedback):
        n = NUMBER_OF_CHANNELS
        cortex = np.asarray(cortex, dtype=float)
        sensors = np.asarray(sensors, dtype=float)
        bhfeedback = np.asarray(bhfeedback, dtype=float)

        # Recurrent inhibition uses last step's d1/d2 outputs (matches the
        # C++ ordering: computed from the member arrays BEFORE this call
        # overwrites them below).
        rec_d1 = np.zeros(n)
        rec_d2 = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    rec_d1[i] -= self.d1_output[j] * W_HSD1
                    rec_d2[i] -= self.d2_output[j] * W_HSD2

        dot_product = SENSORY_WEIGHTS @ sensors
        self.last_dot_product = dot_product   # exposed for diagnostics/logging only

        total_stn = 0.0
        for i in range(n):
            self.stn_output[i] = self.stn[i].output(
                cortex[i], CORTEX_WEIGHTS[i], dot_product[i], self.gpe_output[i], bhfeedback[i])
            total_stn += self.stn_output[i]

        for i in range(n):
            self.d1_output[i] = self.d1[i].output(
                cortex[i], CORTEX_WEIGHTS[i], dot_product[i], bhfeedback[i], rec_d1[i], self.dopamine_d1)
            self.d2_output[i] = self.d2[i].output(
                cortex[i], CORTEX_WEIGHTS[i], dot_product[i], bhfeedback[i], rec_d2[i], self.dopamine_d2)

        output = np.zeros(n)
        for i in range(n):
            self.gpe_output[i] = self.gpe[i].output(self.d2_output[i], total_stn)
            output[i] = self.gpi[i].output(self.d1_output[i], total_stn, self.gpe_output[i])

        return output


# ============================================================
# bg_controller.c :: main()'s thalamocortical loop, hardware stripped out
# ============================================================
class BGController:
    """
    The pure-math part of bg_controller.c's outer for(;;) loop: one call to
    step() = one thalamocortical update (which itself runs the inner
    for(t=0;t<MAXTIME;t+=TIMESTEP) integration -- with the original
    MAXTIME=TIMESTEP=0.2 that inner loop only ever executes once, so this
    matches the original 1:1 by default).

    NEW, from the user's own 2024 paper with Prescott (Biomimetics 9(3):
    139): the real Khepera I hardware ran this circuit "to convergence"
    each decision step, at approximately 7Hz -- not a fixed number of
    inner iterations (MAXTIME=1.2 was an earlier, cruder approximation
    tried this session before that paper was found; convergence is the
    faithful version). Set CONVERGE_MODE=True below (or via
    EASA_CONVERGE=1 in easa_standalone.py) to iterate the inner
    thalamocortical loop until channel_thalamus_out stops changing by
    more than CONVERGE_EPS, instead of a fixed MAXTIME/TIMESTEP count.
    CONVERGE_MAX_ITERS caps it as a safety net in case some input
    combination never settles (a genuine possibility in a system with
    reciprocal inhibition) -- if that cap is ever hit, step() prints a
    one-line warning so it doesn't fail silently.

    Everything about sensing the world, driving motors, or running behaviour
    modules is deliberately NOT here -- that's the next phase, an adapter
    against the PyBullet sim, not a translation of hardware calls.
    """

    CONVERGE_MODE = False
    CONVERGE_EPS = 1e-4
    CONVERGE_MAX_ITERS = 500
    def __init__(self, dopamine_d1=D1_DOPAMINE_DEFAULT, dopamine_d2=D2_DOPAMINE_DEFAULT):
        self.channels = [Channel() for _ in range(NUMBER_OF_CHANNELS)]
        self.bg = BasalGanglia(dopamine_d1, dopamine_d2)
        self.channel_thalamus_out = np.zeros(NUMBER_OF_CHANNELS)

    def step(self, sensors, bhfeedback=None):
        if bhfeedback is None:
            bhfeedback = np.zeros(NUMBER_OF_CHANNELS)
        n = NUMBER_OF_CHANNELS

        def one_inner_iteration():
            channel_cortex_output = np.array([
                self.channels[i].i_salience_cortex(self.channel_thalamus_out[i]) for i in range(n)])

            channel_trn_output = np.array([
                self.channels[i].trn_output(channel_cortex_output[i], self.channel_thalamus_out[i])
                for i in range(n)])

            channel_bg_output = self.bg.output(channel_cortex_output, sensors, bhfeedback)

            rec_trn = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        rec_trn[i] -= channel_trn_output[j] * W_TRNTH

            self.channel_thalamus_out = np.array([
                self.channels[i].thalamus_feedback(
                    channel_cortex_output[i], channel_bg_output[i], rec_trn[i])
                for i in range(n)])
            return channel_bg_output

        if self.CONVERGE_MODE:
            # NEW: compares against TWO iterations back, not one -- this
            # circuit's non-dominant channels settle into a stable
            # PERIOD-2 limit cycle (confirmed empirically: thalamus
            # alternates between two fixed values forever, never landing
            # on one), so a same-iteration-back comparison never
            # converges even at a loose epsilon. Comparing same-phase
            # iterations (n vs n-2) correctly detects when that
            # alternating pattern itself has stabilized, which happens
            # quickly in practice.
            channel_bg_output = np.zeros(n)
            thalamus_two_back = self.channel_thalamus_out.copy()
            thalamus_one_back = self.channel_thalamus_out.copy()
            for iteration in range(self.CONVERGE_MAX_ITERS):
                channel_bg_output = one_inner_iteration()
                if iteration >= 1:
                    if np.max(np.abs(self.channel_thalamus_out - thalamus_two_back)) < self.CONVERGE_EPS:
                        break
                thalamus_two_back = thalamus_one_back
                thalamus_one_back = self.channel_thalamus_out.copy()
            else:
                print(f"[WARN] BGController.step: hit CONVERGE_MAX_ITERS="
                      f"{self.CONVERGE_MAX_ITERS} without converging to "
                      f"CONVERGE_EPS={self.CONVERGE_EPS} (checked against "
                      f"iteration n-2, to allow for the period-2 limit cycle)")
            return channel_bg_output
        else:
            n_inner_steps = max(1, int(round(MAXTIME / TIMESTEP)))
            channel_bg_output = np.zeros(n)
            for _ in range(n_inner_steps):
                channel_bg_output = one_inner_iteration()
            return channel_bg_output