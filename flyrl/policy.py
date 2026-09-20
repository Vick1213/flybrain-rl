"""BrainPolicy: a small trainable encoder/decoder wrapped around a FROZEN
fly-connectome spiking brain (flyrl.fastbrain.FastBrain).

Architecture (v4, Task 1)
--------------------------
obs (B, 12) --[encoder, trainable]--> group_rates_Hz (B, 8), per-group ceiling
  --[broadcast to anatomical input-neuron groups (flyrl.io_neurons, v4:
  ONE shared lateralized steering pair (steer_L/steer_R, visual_projection
  LPLC4) plus 6 unchanged own-channel contact/interoceptive groups)]-->
  neuron_rates (B, K_in)
  --[FastBrain, FROZEN, batch=B, dt=0.5ms, steps_per_action dt-steps]-->
  spikes on the readout neuron set (all descending+motor neurons, see
  scripts/diag_readout.py)
  --[leaky trace, tau=50ms, pooled into 64 pools: 48 named DN cell_type x
  side pools chosen by cross-channel response variance + 16 generic
  side-split pools, z-normalized with fixed screen stats; optionally
  concatenated with a second tau=200ms trace over the SAME 64 pools and/or
  a "no-signal" indicator feature -- see USE_SLOW_TRACE/ADD_NO_SIGNAL_FEATURE
  below]--> features (B, N_FEATURES)
  --[decoder, trainable, anatomical-prior init]--> action = tanh(W_dec @ features + b_dec)  (B, 2)

v4 encoder: valuation lives in the encoder, not in anatomy
-----------------------------------------------------------
v3 gave each of the three directional sources (food/smoke/reels) its OWN
lateralized VPN population (LPLC4/LPC2/LC10e respectively), which produced
a steering-ABILITY imbalance unrelated to preference (food, with the
biggest population, steered as well as the scripted teacher; smoke and
reels, with much smaller populations, steered far worse) -- see the module
docstring of flyrl.io_neurons and results/screen/io_v3_choice.json /
io_v4_choice.json for the full data.

v4 routes ALL THREE sources through the SAME shared pair (steer_L/steer_R,
full LPLC4 populations) and makes the trainable encoder responsible for
combining each source's L/R obs intensity, the L/R bilateral CONTRAST, and
internal state (hunger/nicotine/withdrawal) into that one pair's drive:

    for side X in {L, R}, sign s_X = +1 (L) / -1 (R):
      drive_X = b + sum_src [ a_src * I_src,X
                               + s_X * c_src * C_src
                               + s_X * sum_state m_src,state * state * C_src ]
                  + sum_state h_state * state
      rate_X = max_rate_steer * sigmoid(drive_X)

    where src in {food, smoke, reels}, state in {hunger, nicotine,
    withdrawal}, I_src,X is that source's side-X obs intensity, and
    C_src = (L_src - R_src) / (L_src + R_src + eps) is the bilateral
    contrast (0 when both sides read 0).

This is mirror-symmetric BY CONSTRUCTION: swapping every source's L/R obs
values swaps (I_src,L <-> I_src,R) and negates every C_src, which swaps
drive_L and drive_R exactly (same a/c/m/h/b weights are shared by both
sides -- there is no separate "left encoder" and "right encoder"). A
negative c_src makes rate_X respond LESS to a source on X's own side (and
more to it on the opposite side), i.e. steers away from that source; m
lets internal state modulate a source's attractiveness (e.g. m_food,hunger
> 0 makes food more attractive when hungry). This is exactly where
addiction/preference will live once ES trains these weights (Task 3).

v4.1 encoder: intensity-weighted contrast (fixes 3-source averaging)
----------------------------------------------------------------------
v4's C_src = (L-R)/(L+R+eps) is DISTANCE-INDEPENDENT: a faint, far-away
source's contrast is just as large as a near, strong one's. With all three
sources present (the full addiction arena), drive_X becomes the SUM of
three roughly-unit-scale contrast terms and the fly steers toward their
average direction instead of committing to the nearest/strongest one --
diagnosed from two hijacked ES runs (results/addicted_v4_lr003,
results/addicted_v4_gamma098) where preferences drifted correctly toward
smoke but realised time-at-source stayed ~5-10% and frac_smoke was
unstable across generations (see this module's CLAUDE.md-directed task
spec for the run numbers).

v4.1 fixes this by weighting every C_src by an intensity factor
W_src = sqrt((L_src + R_src) / 2) (0 when both sides read 0, monotonic in
distance since both L and R decay with distance -- see
FlyAddictionEnv.odor_lambda), so a far/faint source's contrast contributes
much less to the drive than a near/strong one, letting the fly commit to
whichever source it is actually close to instead of averaging bearings:

    for side X in {L, R}, sign s_X = +1 (L) / -1 (R):
      drive_X = b + sum_src [ a_src * I_src,X
                               + s_X * c_src * (W_src * C_src)
                               + s_X * sum_state m_src,state * state * (W_src * C_src) ]
                  + sum_state h_state * state

Only the c and m terms (which use C_src) are reweighted; the a_src * I_src
term is untouched (it was never distance-blind -- I_src,X already IS the
raw per-side intensity). Mirror symmetry is preserved: W_src depends only
on L+R (symmetric under an L/R swap), so swapping every source's L/R obs
still swaps drive_L and drive_R exactly, same as v4.

The exponent is configurable via BrainPolicy's `contrast_weighting`
constructor argument so v4's pure-contrast behaviour stays reproducible
behind a flag instead of being silently lost:
  'none'    -- W_src == 1 always (the original v4 formula, unweighted).
  'sqrt'    -- W_src = sqrt((L+R)/2)  (v4.1 default).
  'quarter' -- W_src = ((L+R)/2)**0.25 (gentler falloff, tried if 'sqrt'
              weakens far-range steering enough to drop DAgger reach below
              the Task B gate -- see results/taxis_dagger_v41/).
This changes ONLY the runtime encoder formula (BrainPolicy._encode), never
the 19-param flat layout or parameter names -- theta vectors are fully
interchangeable across weighting modes.

The remaining six anatomical groups (sugar_taste, nicotine_taste,
reels_jackpot, hunger, nicotine, withdrawal) keep the simple v1-v3
own-channel form: rate = max_rate * sigmoid(w * obs_channel + b).

Every parameter of the encoder+decoder lives in a single flat vector per
batch element: BrainPolicy is BATCHED, i.e. parameters are a (B, P) tensor
-- one full encoder+decoder per batch element (used by ES: one population
member per batch row), all sharing ONE FastBrain(batch=B) instance (the
brain weights themselves are never trained/mutated). Flat layout (fixed,
see set_params): the first N_STEER_PARAMS (19, or 22 if
ADD_STOP_ON_CONTACT -- see that flag's docstring) entries are ALWAYS the
steering block (b, a[3], c[3], m[3x3], h[3], optionally g[3], in that
order), followed by N_CONTACT_PARAMS=12 (w,b per contact group), followed
by the decoder (W_dec, b_dec). flyrl.train_es relies on this fixed layout
to give the steering params a different (larger) ES mutation sigma than
the rest.

Anatomical input groups and the readout neuron set are fixed, deterministic
index sets built by flyrl.io_neurons (input groups, cached to
flyrl/io_neurons.npz) and scripts/diag_readout.py (readout set + 64-pool
assignment + DAN indices for logging, cached to flyrl/readout_neurons.npz).
BrainPolicy requires scripts/diag_readout.py to have been run at least once
before it can be constructed (see _load_readout_cache below).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from flyrl.fastbrain import FastBrain
from flyrl.io_neurons import GROUP_NAMES, GROUP_MAX_RATE_HZ, build_input_groups, group_sizes, union_and_group_ids

READOUT_CACHE_PATH = Path(__file__).resolve().parent / "readout_neurons.npz"

N_GROUPS = len(GROUP_NAMES)          # 8 anatomical input groups (v4): steer_L, steer_R, + 6 contact/intero
N_POOLS = 64                          # anatomical DN/motor readout pools (48 named + 16 generic), unchanged
N_ACTIONS = 2

# ---------------------------------------------------------------------
# v4 steering block: obs-channel indices (fixed by the FROZEN
# FlyAddictionEnv.obs_channels order) for the three bilateral source pairs,
# the three internal-state channels, and the six own-channel contact groups.
# ---------------------------------------------------------------------
_STEER_SOURCES = ("food", "smoke", "reels")
_STEER_STATES = ("hunger", "nicotine", "withdrawal")
_BILATERAL_OBS_PAIRS = [(0, 1), (2, 3), (4, 5)]  # (food_L,R), (smoke_L,R), (reels_L,R) obs-channel indices
_STATE_OBS_IDX = [9, 10, 11]                      # hunger, nicotine, withdrawal obs-channel indices
# Contact/interoceptive groups, in GROUP_NAMES[2:] order, with their own obs-channel index.
_CONTACT_GROUP_NAMES = GROUP_NAMES[2:]             # ["sugar_taste","nicotine_taste","reels_jackpot","hunger","nicotine","withdrawal"]
_CONTACT_OBS_IDX = [6, 7, 8, 9, 10, 11]
assert len(_CONTACT_GROUP_NAMES) == len(_CONTACT_OBS_IDX)

N_STEER_SOURCES = len(_STEER_SOURCES)   # 3
N_STEER_STATES = len(_STEER_STATES)     # 3
N_CONTACT_GROUPS = len(_CONTACT_OBS_IDX)  # 6

# ---------------------------------------------------------------------
# Task C (commitment check): OPTIONAL motor-side "stop-on-contact" reflex.
# This is NOT part of the connectome-driven encoder -- it post-processes
# the DECODER's own forward-speed output using the raw taste/jackpot obs
# channels (sugar_taste, nicotine_taste, reels_jackpot), exactly like a
# reflex arc that bypasses the brain: forward_action = decoder_forward *
# (1 - sum_k g_k * taste_k), clipped to [-1, 1]. Added ONLY if
# results/commitment_check found short dwell times (fly walks through a
# source instead of stopping) -- see that report for the empirical
# decision. When True, the 3 trainable gains g_sugar/g_nicotine/g_reels are
# APPENDED to the steering param block (steering_param_names order:
# ..., h_hunger, h_nicotine, h_withdrawal, g_sugar, g_nicotine, g_reels),
# so N_STEER_PARAMS becomes 22 and N_PARAMS grows by 3 accordingly -- every
# helper below (steering_param_names/unpack/default_params/set_params)
# keys off this one flag, no half-measures.
# DECISION (results/commitment_check/report.json, 16 fixed seeds x 300
# steps, FULL env, DAgger decoder from results/taxis_dagger_v4 (v4, 'none'
# weighting) and results/taxis_dagger_v41 (v4.1, 'sqrt' weighting), init
# encoder): mean dwell length was 166-191 steps (NOT short) for BOTH
# variants, and the SIGNED mean forward action while at a source was
# strongly negative (v4: -0.97, v4.1: -0.98, i.e. near-ZERO actual forward
# speed) -- the DAgger-fitted decoder already learned to hover on contact
# purely by imitating the scripted teacher's own hover behaviour (see
# flyrl.scripted._steer_towards: speed_action=-1.0 when info['at'] ==
# target). Not enabled.
ADD_STOP_ON_CONTACT = False
_TASTE_PARAM_NAMES = ("sugar_taste", "nicotine_taste", "reels_jackpot")
_TASTE_OBS_IDX = [6, 7, 8]  # sugar_taste, nicotine_taste, reels_jackpot obs-channel indices
_STEER_INIT_G = 0.8
_N_STOP_ON_CONTACT_PARAMS = len(_TASTE_OBS_IDX)

# Flat steering-block layout: b(1), a(3), c(3), m(3x3=9), h(3) = 19, plus
# an optional g(3) stop-on-contact block (see ADD_STOP_ON_CONTACT) = 22.
N_STEER_PARAMS = (1 + N_STEER_SOURCES + N_STEER_SOURCES + N_STEER_SOURCES * N_STEER_STATES + N_STEER_STATES
                   + (_N_STOP_ON_CONTACT_PARAMS if ADD_STOP_ON_CONTACT else 0))
assert N_STEER_PARAMS == (22 if ADD_STOP_ON_CONTACT else 19)
N_CONTACT_PARAMS = 2 * N_CONTACT_GROUPS  # w,b per contact group = 12
assert N_CONTACT_PARAMS == 12
N_ENC_PARAMS = N_STEER_PARAMS + N_CONTACT_PARAMS  # 31

_CONTRAST_EPS = 1e-6
_NO_SIGNAL_EPS = 1e-3  # matches flyrl.scripted's own lost-signal-eps convention
_STEER_INIT_A = 1.5
_STEER_INIT_C = 4.0  # v4.1 (was 3.0 in v4) -- see module docstring
_STEER_INIT_B = -2.0
_DIAG_INIT = 4.0
_BIAS_INIT = -3.0

# v4.1 intensity-weighted contrast (see module docstring): W_src =
# ((L_src+R_src)/2) ** power, power=None means W_src == 1 (v4's original,
# unweighted contrast, kept reproducible behind contrast_weighting='none').
CONTRAST_WEIGHTING_POWERS = {"none": None, "sqrt": 0.5, "quarter": 0.25}
DEFAULT_CONTRAST_WEIGHTING = "sqrt"

# v2: default anatomical-prior decoder init scale (see default_params below
# and results/screen/io_v2_choice.json "encoder_design_decision"). ES trains
# all weights from here; this only sets the untrained starting point.
DEFAULT_INIT_SCALE = 0.3

TAU_TRACE_MS = 50.0

# Task 2 (DAgger, flyrl.dagger_taxis) latency ablation: an OPTIONAL second,
# slower leaky trace over the same readout spikes, maintained alongside the
# tau=50ms trace above at negligible extra cost (one more elementwise
# multiply-add per sub-step). The RAW tau=200ms trace is always computed
# and exposed via features_slow_raw() for flyrl.dagger_taxis's own ablation
# check; whether it is actually CONCATENATED into the decoder's own input
# (doubling the feature dim to 128) is controlled by USE_SLOW_TRACE below.
TAU_TRACE_SLOW_MS = 200.0

# ---------------------------------------------------------------------
# Task 1/2 decisions (empirically set from flyrl.dagger_taxis's v4 run,
# results/taxis_dagger_v4/controls.json): whether the tau=200ms trace and a
# "no external signal" indicator feature are concatenated into the
# decoder's actual input. Flipping either flag changes N_FEATURES/N_PARAMS
# globally -- see the module docstring's "no half-measures" requirement:
# set_params, n_params, default_params, --init-from and evaluate all key
# off these two flags.
#
# USE_SLOW_TRACE = True: the v4 DAgger ablation (pass 1, 64-dim-only decoder)
# measured mean val R^2 = 0.267 (fast-only) vs 0.373 (fast+slow concatenated,
# both z-normed via FIXED screen stats) -- a +0.106 improvement, well over
# the spec's >0.03 bar. Adopted: feature dim is now 128 (64 fast + 64 slow),
# decoder 2x128+2. flyrl.dagger_taxis was then RE-RUN from scratch (not
# patched) to fit the real 128-dim decoder -- see results/taxis_dagger_v4/.
#
# ADD_NO_SIGNAL_FEATURE = False: the same pass-1 run's no-signal diagnostic
# (results/taxis_dagger_v4/controls.json's "no_signal_search_diagnostic",
# from flyrl.dagger_taxis.diagnose_no_signal_search) found reels ALREADY
# reached 100% at 300 steps and already turns MORE when it has no signal
# (mean |turn|=0.42) than when it does (mean |turn|=0.37) -- i.e. it already
# searches on its own, decoded from the existing 64 fast-trace pools. The
# spec's trigger condition ("if reels reach lags the others mainly for that
# reason") did not hold, so this feature was NOT added.
# ---------------------------------------------------------------------
USE_SLOW_TRACE = True
ADD_NO_SIGNAL_FEATURE = False

N_FEATURE_POOLS = N_POOLS * (2 if USE_SLOW_TRACE else 1)
N_EXTRA_FEATURES = 1 if ADD_NO_SIGNAL_FEATURE else 0
N_FEATURES = N_FEATURE_POOLS + N_EXTRA_FEATURES

# Per-group encoder output ceiling in Hz, decided empirically per
# anatomical group by the Task A/v3 screen (see
# flyrl.io_neurons.GROUP_MAX_RATE_HZ / results/screen/io_v3_choice.json)
# so each group's operating range avoids network ignition.
_GROUP_MAX_RATE_ARRAY = np.array([GROUP_MAX_RATE_HZ[name] for name in GROUP_NAMES], dtype=np.float32)

# Flat-parameter layout (all sizes fixed once N_POOLS/N_FEATURES are fixed).
_N_DEC_W = N_ACTIONS * N_FEATURES
_N_DEC_B = N_ACTIONS
N_PARAMS = N_ENC_PARAMS + _N_DEC_W + _N_DEC_B


def _load_readout_cache():
    if not READOUT_CACHE_PATH.exists():
        raise RuntimeError(
            f"{READOUT_CACHE_PATH} not found. Run `python scripts/diag_readout.py` "
            "first -- it decides and caches the readout neuron set (see spec: "
            "'FIRST run a diagnostic script')."
        )
    data = np.load(READOUT_CACHE_PATH)
    pool_mean_slow = data["pool_mean_slow"].astype(np.float32) if "pool_mean_slow" in data else None
    pool_std_slow = data["pool_std_slow"].astype(np.float32) if "pool_std_slow" in data else None
    return (
        data["readout_idx"].astype(np.int64),
        data["pool_assign"].astype(np.int64),
        data["dan_idx"].astype(np.int64),
        int(data["n_pools"]),
        data["pool_mean"].astype(np.float32),
        data["pool_std"].astype(np.float32),
        data["pool_turn_sign"].astype(np.float32),
        pool_mean_slow,
        pool_std_slow,
    )


def steering_param_names() -> list:
    """Names of the N_STEER_PARAMS steering-block entries, in flat-theta
    order (see set_params): b, a_food/a_smoke/a_reels, c_food/c_smoke/
    c_reels, m_<src>_<state> (9, src-major), h_hunger/h_nicotine/h_withdrawal,
    and -- ONLY if ADD_STOP_ON_CONTACT -- g_sugar_taste/g_nicotine_taste/
    g_reels_jackpot (19 -> 22 entries). Used by flyrl.train_es to log
    preference drift (Task 3)."""
    names = ["b_steer"]
    names += [f"a_{src}" for src in _STEER_SOURCES]
    names += [f"c_{src}" for src in _STEER_SOURCES]
    names += [f"m_{src}_{state}" for src in _STEER_SOURCES for state in _STEER_STATES]
    names += [f"h_{state}" for state in _STEER_STATES]
    if ADD_STOP_ON_CONTACT:
        names += [f"g_{name}" for name in _TASTE_PARAM_NAMES]
    assert len(names) == N_STEER_PARAMS
    return names


def unpack_steering_params(theta: np.ndarray) -> dict:
    """theta: (N_PARAMS,) or longer flat vector (only the first
    N_STEER_PARAMS entries are used) -> {name: float value} for every
    steering-block parameter (see steering_param_names)."""
    theta = np.asarray(theta)
    names = steering_param_names()
    return {name: float(theta[i]) for i, name in enumerate(names)}


def steering_param_sigma_vector(sigma_base: float, sigma_steer: float, n_params: int = N_PARAMS) -> np.ndarray:
    """(n_params,) per-parameter ES mutation sigma: sigma_steer for the
    first N_STEER_PARAMS entries (the 19 steering-encoder params -- O(1)
    quantities where preference lives, spec: larger sigma), sigma_base for
    everything else (the 12 contact/intero encoder params + decoder,
    spec: smaller sigma). Relies on flyrl.policy's FIXED flat-theta layout
    (steering block always first) -- see set_params."""
    vec = np.full(n_params, float(sigma_base), dtype=np.float64)
    vec[:N_STEER_PARAMS] = float(sigma_steer)
    return vec


def default_params(seed: int = 0, init_scale: float = DEFAULT_INIT_SCALE,
                    init_c: float = _STEER_INIT_C) -> np.ndarray:
    """The spec-prescribed v4.1 initialization, flattened to a (N_PARAMS,)
    vector.

    Steering block (19, or 22 if ADD_STOP_ON_CONTACT): a=+1.5, c=init_c
    (default +4.0, v4.1 -- was +3.0 in v4), m=0, h=0, b=-2 for all three
    sources -- i.e. EQUAL innate attraction to food/smoke/reels (m=0: no
    state-dependent modulation yet), matching the spec's "equal innate
    attraction to all three sources" starting point. `init_c` is exposed
    so callers (flyrl.dagger_taxis's --init-c) can raise it (e.g. to 6) if
    v4.1's intensity weighting weakens far-range steering enough to hurt
    DAgger reach. If ADD_STOP_ON_CONTACT, the 3 stop-on-contact gains
    (g_sugar_taste/g_nicotine_taste/g_reels_jackpot) are appended at
    _STEER_INIT_G (0.8) each.

    Contact/interoceptive block (12 params, 6 groups): w=+4, b=-3 (own-
    channel drive), unchanged from v1-v3.

    Decoder: ANATOMICAL-PRIOR init (v2, Task B) reused as-is: the turn
    output's weight on each of the 64 (tau=50ms) readout-pool features gets
    sign = pool_turn_sign (from the screen: +1 if that DN/motor pool
    responds more to LEFT-side steering input, -1 if more to RIGHT),
    scaled by `init_scale`; if USE_SLOW_TRACE, the 64 tau=200ms pool
    features get the SAME anatomical prior (same pools, just a slower
    filter -- no reason to expect the sign to flip); any additional
    "no-signal" feature gets a small-random weight like the rest of the
    forward-speed row. Forward-speed output keeps a small-random weight +
    bias=+0.5. ES trains every weight from here; `init_scale` only sets the
    starting point."""
    rng = np.random.default_rng(seed)

    b_steer = np.array([_STEER_INIT_B], dtype=np.float32)
    a_steer = np.full(N_STEER_SOURCES, _STEER_INIT_A, dtype=np.float32)
    c_steer = np.full(N_STEER_SOURCES, init_c, dtype=np.float32)
    m_steer = np.zeros((N_STEER_SOURCES, N_STEER_STATES), dtype=np.float32)
    h_steer = np.zeros(N_STEER_STATES, dtype=np.float32)
    steer_parts = [b_steer, a_steer, c_steer, m_steer.reshape(-1), h_steer]
    if ADD_STOP_ON_CONTACT:
        g_steer = np.full(_N_STOP_ON_CONTACT_PARAMS, _STEER_INIT_G, dtype=np.float32)
        steer_parts.append(g_steer)
    steer_block = np.concatenate(steer_parts)
    assert steer_block.shape == (N_STEER_PARAMS,)

    w_contact = np.full(N_CONTACT_GROUPS, _DIAG_INIT, dtype=np.float32)
    b_contact = np.full(N_CONTACT_GROUPS, _BIAS_INIT, dtype=np.float32)
    contact_block = np.concatenate([w_contact, b_contact])
    assert contact_block.shape == (N_CONTACT_PARAMS,)

    W_dec = (rng.standard_normal((N_ACTIONS, N_FEATURES)) * 0.05).astype(np.float32)
    try:
        _, _, _, _, _, _, pool_turn_sign, _, _ = _load_readout_cache()
        W_dec[0, :N_POOLS] = init_scale * pool_turn_sign  # turn output: anatomical-prior sign (fast trace)
        if USE_SLOW_TRACE:
            W_dec[0, N_POOLS:2 * N_POOLS] = init_scale * pool_turn_sign  # same prior, slow trace
    except RuntimeError:
        pass  # readout cache not built yet (e.g. diag_readout.py hasn't run) -- fall back to random
    b_dec = np.array([0.0, 0.5], dtype=np.float32)

    theta = np.concatenate([
        steer_block, contact_block, W_dec.reshape(-1), b_dec.reshape(-1),
    ]).astype(np.float32)
    assert theta.shape == (N_PARAMS,)
    return theta


class BrainPolicy:
    """Batched trainable encoder/decoder around a frozen FastBrain.

    Parameters
    ----------
    batch : int
        Number of independent (brain, encoder, decoder) instances, all
        sharing one FastBrain(batch=batch). Each batch row has its own
        encoder+decoder parameters (set via set_params).
    device : str
        'cpu' or 'mps', forwarded to FastBrain.
    dt : float
        Brain simulation timestep in ms (spec: use 0.5 for training).
    steps_per_action : int
        Number of dt-steps simulated per env.step() (spec: 20, i.e. 10ms).
    seed : int, optional
        FastBrain's own Poisson-sampling RNG seed.
    contrast_weighting : str
        v4.1 (see module docstring): how each source's bilateral contrast
        C_src is weighted by intensity before entering the steering drive.
        One of CONTRAST_WEIGHTING_POWERS' keys -- 'none' (v4's original,
        unweighted contrast), 'sqrt' (v4.1 default), 'quarter'.
    """

    def __init__(self, batch: int, device: str = "cpu", dt: float = 0.5,
                 steps_per_action: int = 20, seed: int | None = None,
                 contrast_weighting: str = DEFAULT_CONTRAST_WEIGHTING):
        self.batch = int(batch)
        self.device = torch.device(device)
        self.dt = float(dt)
        self.steps_per_action = int(steps_per_action)
        if contrast_weighting not in CONTRAST_WEIGHTING_POWERS:
            raise ValueError(
                f"contrast_weighting must be one of {sorted(CONTRAST_WEIGHTING_POWERS)}, "
                f"got {contrast_weighting!r}"
            )
        self.contrast_weighting = contrast_weighting
        self._contrast_weight_power = CONTRAST_WEIGHTING_POWERS[contrast_weighting]

        self.fb = FastBrain(batch=self.batch, device=device, dt=dt, seed=seed)

        self.groups = build_input_groups(self.fb)
        self.group_sizes = group_sizes(self.groups)
        union_idx, group_id_per_neuron = union_and_group_ids(self.groups, GROUP_NAMES)
        self.fb.set_input_neurons(union_idx)
        # Input/sensory-proxy neurons should not be refractory-gated, matching
        # how the reference benchmark treats externally-driven neurons (see
        # FastBrain.set_exc_indices docstring / eon-fly-brain sugar experiment).
        self.fb.set_exc_indices(union_idx)
        self.n_input_neurons = int(len(union_idx))
        self._group_id_per_neuron = torch.as_tensor(group_id_per_neuron, dtype=torch.long, device=self.device)

        (readout_idx, pool_assign, dan_idx, n_pools, pool_mean, pool_std, pool_turn_sign,
         pool_mean_slow, pool_std_slow) = _load_readout_cache()
        assert n_pools == N_POOLS, f"cached readout was built with n_pools={n_pools}, expected {N_POOLS}"
        self.readout_idx = torch.as_tensor(readout_idx, dtype=torch.long, device=self.device)
        self.pool_assign = torch.as_tensor(pool_assign, dtype=torch.long, device=self.device)
        self.dan_idx = torch.as_tensor(dan_idx, dtype=torch.long, device=self.device)
        self.n_readout = int(len(readout_idx))
        self.n_dan = int(len(dan_idx))
        # v2: fixed z-normalization stats for the 64 readout-pool features,
        # computed once from the Task A/B screen (scripts/diag_readout.py)
        # and never updated online (spec: "z-normalised with fixed stats
        # from the screen").
        self.pool_mean = torch.as_tensor(pool_mean, dtype=torch.float32, device=self.device)
        self.pool_std = torch.as_tensor(pool_std, dtype=torch.float32, device=self.device)
        self.pool_turn_sign = torch.as_tensor(pool_turn_sign, dtype=torch.float32, device=self.device)
        if USE_SLOW_TRACE:
            assert pool_mean_slow is not None and pool_std_slow is not None, (
                "USE_SLOW_TRACE=True but flyrl/readout_neurons.npz has no pool_mean_slow/"
                "pool_std_slow -- rerun scripts/diag_readout.py (v4) to rebuild the cache."
            )
            self.pool_mean_slow = torch.as_tensor(pool_mean_slow, dtype=torch.float32, device=self.device)
            self.pool_std_slow = torch.as_tensor(pool_std_slow, dtype=torch.float32, device=self.device)

        # DAN logging split (Task 1): DAN neurons that ARE anatomical input
        # neurons (the "nicotine" PAM group and "withdrawal" PPL group are
        # both directly-driven subsets of the full DAN population) vs DAN
        # neurons that are NOT driven by any input group. mean_dan_rate_hz()
        # reports the latter only; dan_rate_driven_hz() the former.
        union_idx_set = set(int(i) for i in union_idx.tolist())
        dan_driven_mask_np = np.array([int(i) in union_idx_set for i in dan_idx], dtype=bool)
        self._dan_driven_mask = torch.as_tensor(dan_driven_mask_np, dtype=torch.bool, device=self.device)
        self.n_dan_driven = int(dan_driven_mask_np.sum())
        self.n_dan_other = int((~dan_driven_mask_np).sum())

        self.n_groups = N_GROUPS
        self.n_pools = N_POOLS
        self.n_features = N_FEATURES
        self.n_actions = N_ACTIONS
        self.n_params = N_PARAMS

        # Per-group encoder rate ceiling (Hz), decided empirically per
        # anatomical group by the Task A screen (see
        # flyrl.io_neurons.GROUP_MAX_RATE_HZ) instead of v1's single
        # constant, so each group's operating range avoids network ignition.
        self.group_max_rate = torch.as_tensor(_GROUP_MAX_RATE_ARRAY, dtype=torch.float32, device=self.device)

        # Leaky trace (tau=50ms) decay per dt sub-step.
        self.decay = float(np.exp(-self.dt / TAU_TRACE_MS))
        # Task 2: second, slower trace (tau=200ms) -- see
        # TAU_TRACE_SLOW_MS docstring above. Always maintained (cheap); only
        # CONCATENATED into the decoder input if USE_SLOW_TRACE.
        self.decay_slow = float(np.exp(-self.dt / TAU_TRACE_SLOW_MS))

        self.theta = None
        self.set_params(np.tile(default_params(), (self.batch, 1)))
        self.reset()

    # ------------------------------------------------------------------
    def set_params(self, theta):
        """theta: (B, N_PARAMS) array-like (numpy or torch). Parameters of
        batch row i only ever affect BrainPolicy.act's output for batch
        row i (no cross-batch mixing anywhere in encoder/decoder). Flat
        layout (see module docstring): steering block (19, or 22 if
        ADD_STOP_ON_CONTACT) first, then the 6-group contact/intero block
        (12), then the decoder (W_dec, b_dec).
        """
        theta = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
        assert theta.shape == (self.batch, self.n_params), (
            f"expected theta shape ({self.batch}, {self.n_params}), got {tuple(theta.shape)}"
        )
        self.theta = theta
        off = 0
        self.b_steer = theta[:, off:off + 1]
        off += 1
        self.a_steer = theta[:, off:off + N_STEER_SOURCES]
        off += N_STEER_SOURCES
        self.c_steer = theta[:, off:off + N_STEER_SOURCES]
        off += N_STEER_SOURCES
        self.m_steer = theta[:, off:off + N_STEER_SOURCES * N_STEER_STATES].reshape(
            self.batch, N_STEER_SOURCES, N_STEER_STATES)
        off += N_STEER_SOURCES * N_STEER_STATES
        self.h_steer = theta[:, off:off + N_STEER_STATES]
        off += N_STEER_STATES
        if ADD_STOP_ON_CONTACT:
            self.g_stop = theta[:, off:off + _N_STOP_ON_CONTACT_PARAMS]
            off += _N_STOP_ON_CONTACT_PARAMS
        else:
            self.g_stop = None
        assert off == N_STEER_PARAMS

        self.w_contact = theta[:, off:off + N_CONTACT_GROUPS]
        off += N_CONTACT_GROUPS
        self.b_contact = theta[:, off:off + N_CONTACT_GROUPS]
        off += N_CONTACT_GROUPS
        assert off == N_ENC_PARAMS

        self.W_dec = theta[:, off:off + _N_DEC_W].reshape(self.batch, N_ACTIONS, N_FEATURES)
        off += _N_DEC_W
        self.b_dec = theta[:, off:off + _N_DEC_B].reshape(self.batch, N_ACTIONS)
        off += _N_DEC_B
        assert off == self.n_params

    def reset(self):
        """Reset brain state and readout trace at episode start."""
        self.fb.reset()
        self.trace = torch.zeros((self.batch, self.n_readout), dtype=torch.float32, device=self.device)
        self.trace_slow = torch.zeros((self.batch, self.n_readout), dtype=torch.float32, device=self.device)
        self.last_dan_rate_hz = torch.zeros((self.batch,), dtype=torch.float32, device=self.device)
        self.last_dan_rate_driven_hz = torch.zeros((self.batch,), dtype=torch.float32, device=self.device)
        self.last_pooled = torch.zeros((self.batch, N_FEATURES), dtype=torch.float32, device=self.device)
        self.last_pooled_raw = torch.zeros((self.batch, N_POOLS), dtype=torch.float32, device=self.device)
        self.last_pooled_slow_raw = torch.zeros((self.batch, N_POOLS), dtype=torch.float32, device=self.device)
        self.last_no_signal = torch.zeros((self.batch, 1), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    def _encode(self, obs_t: torch.Tensor):
        """obs_t: (B, 12) -> (group_rates_Hz (B, 8), no_signal (B, 1)). Pure
        function of obs and the current encoder params; does not touch
        brain state. group_rates order matches flyrl.io_neurons.GROUP_NAMES
        (v4): [steer_L, steer_R, sugar_taste, nicotine_taste, reels_jackpot,
        hunger, nicotine, withdrawal]."""
        li = [p[0] for p in _BILATERAL_OBS_PAIRS]
        ri = [p[1] for p in _BILATERAL_OBS_PAIRS]
        L = obs_t[:, li]  # (B, 3): food_L, smoke_L, reels_L
        R = obs_t[:, ri]  # (B, 3): food_R, smoke_R, reels_R
        C = (L - R) / (L + R + _CONTRAST_EPS)  # (B, 3) bilateral contrast per source
        state = obs_t[:, _STATE_OBS_IDX]  # (B, 3): hunger, nicotine, withdrawal

        # v4.1: weight each source's contrast by its own intensity so a
        # far/faint source's contrast contributes less than a near/strong
        # one's (fixes the 3-source bearing-averaging failure mode -- see
        # module docstring). contrast_weighting='none' recovers v4 exactly
        # (W == 1 always).
        if self._contrast_weight_power is None:
            WC = C
        else:
            W = torch.clamp((L + R) * 0.5, min=0.0) ** self._contrast_weight_power  # (B, 3), 0 when L=R=0
            WC = W * C

        term_a_L = (self.a_steer * L).sum(dim=1)   # (B,)
        term_a_R = (self.a_steer * R).sum(dim=1)   # (B,)
        term_c = (self.c_steer * WC).sum(dim=1)    # (B,)
        inner = torch.einsum("bks,bs->bk", self.m_steer, state)  # (B, 3) over src k
        term_m = (inner * WC).sum(dim=1)           # (B,)
        term_h = (self.h_steer * state).sum(dim=1)  # (B,)
        b = self.b_steer.squeeze(1)                # (B,)

        drive_L = b + term_a_L + term_c + term_m + term_h
        drive_R = b + term_a_R - term_c - term_m + term_h

        rate_steer_L = self.group_max_rate[0] * torch.sigmoid(drive_L)  # (B,)
        rate_steer_R = self.group_max_rate[1] * torch.sigmoid(drive_R)  # (B,)

        obs_contact = obs_t[:, _CONTACT_OBS_IDX]                         # (B, 6)
        pre_contact = self.w_contact * obs_contact + self.b_contact      # (B, 6)
        rate_contact = self.group_max_rate[2:].unsqueeze(0) * torch.sigmoid(pre_contact)  # (B, 6)

        group_rates = torch.cat(
            [rate_steer_L.unsqueeze(1), rate_steer_R.unsqueeze(1), rate_contact], dim=1)  # (B, 8)

        bilateral_sum = obs_t[:, [p for pair in _BILATERAL_OBS_PAIRS for p in pair]].sum(dim=1, keepdim=True)
        no_signal = (bilateral_sum < _NO_SIGNAL_EPS).to(torch.float32)  # (B, 1)

        return group_rates, no_signal

    @torch.no_grad()
    def group_rates(self, obs) -> np.ndarray:
        """Encoder output group_rates_Hz (B, n_groups) for `obs`, WITHOUT
        stepping the brain (diagnostics/tests only)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        assert obs_t.shape == (self.batch, 12), f"expected obs shape ({self.batch}, 12), got {tuple(obs_t.shape)}"
        group_rates, _no_signal = self._encode(obs_t)
        return group_rates.cpu().numpy()

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        """obs: (B, 12) array-like in [0,1] (FlyAddictionEnv.obs_channels
        order). Returns actions (B, 2) numpy array in [-1, 1]."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        assert obs_t.shape == (self.batch, 12), (
            f"expected obs shape ({self.batch}, 12), got {tuple(obs_t.shape)}"
        )

        group_rates, no_signal = self._encode(obs_t)  # (B, 8) Hz, (B, 1)
        neuron_rates = group_rates[:, self._group_id_per_neuron]  # (B, K_in)
        self.last_no_signal = no_signal

        dan_spike_sum = torch.zeros((self.batch,), dtype=torch.float32, device=self.device)
        dan_spike_sum_driven = torch.zeros((self.batch,), dtype=torch.float32, device=self.device)
        for _ in range(self.steps_per_action):
            spike = self.fb.step(rates=neuron_rates)  # (B, N) bool
            spike_f = spike.to(torch.float32)
            readout_spikes = spike_f.index_select(1, self.readout_idx)  # (B, n_readout)
            self.trace = self.trace * self.decay + readout_spikes
            self.trace_slow = self.trace_slow * self.decay_slow + readout_spikes
            dan_spikes = spike_f.index_select(1, self.dan_idx)  # (B, n_dan)
            dan_spike_sum += dan_spikes[:, ~self._dan_driven_mask].sum(dim=1)
            dan_spike_sum_driven += dan_spikes[:, self._dan_driven_mask].sum(dim=1)

        window_s = self.steps_per_action * self.dt / 1000.0
        self.last_dan_rate_hz = dan_spike_sum / max(self.n_dan_other, 1) / window_s
        self.last_dan_rate_driven_hz = dan_spike_sum_driven / max(self.n_dan_driven, 1) / window_s

        pooled = torch.zeros((self.batch, N_POOLS), dtype=torch.float32, device=self.device)
        pooled.index_add_(1, self.pool_assign, self.trace)  # (B, 64) raw per-pool trace sum
        # v2: fixed z-normalization from the screen (spec), replacing v1's
        # /pool_counts/trace_norm_const heuristic normalization.
        pooled_normed = (pooled - self.pool_mean.unsqueeze(0)) / self.pool_std.unsqueeze(0)

        pooled_slow_raw = torch.zeros((self.batch, N_POOLS), dtype=torch.float32, device=self.device)
        pooled_slow_raw.index_add_(1, self.pool_assign, self.trace_slow)
        self.last_pooled_raw = pooled
        self.last_pooled_slow_raw = pooled_slow_raw

        feat_parts = [pooled_normed]
        if USE_SLOW_TRACE:
            pooled_slow_normed = (pooled_slow_raw - self.pool_mean_slow.unsqueeze(0)) / self.pool_std_slow.unsqueeze(0)
            feat_parts.append(pooled_slow_normed)
        if ADD_NO_SIGNAL_FEATURE:
            feat_parts.append(no_signal)
        features_full = torch.cat(feat_parts, dim=1)  # (B, N_FEATURES) -- the decoder's ACTUAL input
        self.last_pooled = features_full

        pre_dec = torch.einsum("bof,bf->bo", self.W_dec, features_full) + self.b_dec  # (B, 2)
        action = torch.tanh(pre_dec)

        if ADD_STOP_ON_CONTACT:
            # Motor-side reflex (outside the brain, see ADD_STOP_ON_CONTACT
            # docstring): dampen the DECODER's own forward-speed output by
            # (1 - sum_k g_k * taste_k) using the raw taste/jackpot obs
            # channels directly (sugar_taste, nicotine_taste, reels_jackpot),
            # so the fly slows/stops on contact instead of walking through
            # a source. Turn output is untouched.
            taste = obs_t[:, _TASTE_OBS_IDX]  # (B, 3)
            stop_factor = 1.0 - (self.g_stop * taste).sum(dim=1)  # (B,)
            forward = torch.clamp(action[:, 1] * stop_factor, -1.0, 1.0)
            action = torch.stack([action[:, 0], forward], dim=1)

        return action.cpu().numpy()

    def features(self) -> np.ndarray:
        """(B, N_FEATURES) features from the most recent act() call -- i.e.
        exactly what BrainPolicy.act() feeds its own decoder (z-normalized
        tau=50ms pools, optionally concatenated with z-normalized tau=200ms
        pools and/or a no-signal indicator; see USE_SLOW_TRACE /
        ADD_NO_SIGNAL_FEATURE). Used by flyrl.dagger_taxis to fit an
        external ridge-regression decoder onto the same features
        BrainPolicy.set_params' decoder block would consume (Task 2)."""
        return self.last_pooled.cpu().numpy()

    def features_slow_raw(self) -> np.ndarray:
        """(B, 64) RAW (not z-normalized) tau=200ms pooled features from the
        most recent act() call -- ALWAYS computed regardless of
        USE_SLOW_TRACE, for flyrl.dagger_taxis's own ablation check."""
        return self.last_pooled_slow_raw.cpu().numpy()

    def no_signal_feature(self) -> np.ndarray:
        """(B, 1) -- 1.0 where ALL SIX bilateral steering obs channels
        (food/smoke/reels L+R) read below _NO_SIGNAL_EPS (no directional
        cue at all, e.g. reels outside its field of view), else 0.0.
        Computed regardless of ADD_NO_SIGNAL_FEATURE."""
        return self.last_no_signal.cpu().numpy()

    def mean_dan_rate_hz(self) -> np.ndarray:
        """Per-batch-row mean firing rate (Hz) of DAN neurons that are NOT
        directly-driven anatomical input neurons (excludes the "nicotine"
        PAM and "withdrawal" PPL input groups), averaged over the most
        recent act() call's steps_per_action-step window (logging only --
        DAN activity never feeds the decoder). See dan_rate_driven_hz()."""
        return self.last_dan_rate_hz.cpu().numpy()

    def dan_rate_driven_hz(self) -> np.ndarray:
        """Per-batch-row mean firing rate (Hz) of the directly-driven DAN
        input neurons only (the "nicotine" PAM group + "withdrawal" PPL
        group), averaged over the most recent act() call's window."""
        return self.last_dan_rate_driven_hz.cpu().numpy()
