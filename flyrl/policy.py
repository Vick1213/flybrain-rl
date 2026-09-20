"""BrainPolicy: a small trainable encoder/decoder wrapped around a FROZEN
fly-connectome spiking brain (flyrl.fastbrain.FastBrain).

Architecture
------------
obs (B, 12) --[encoder, trainable]--> group_rates_Hz (B, 12), per-group ceiling
  --[broadcast to anatomical input-neuron groups (flyrl.io_neurons, v2)]--> neuron_rates (B, K_in)
  --[FastBrain, FROZEN, batch=B, dt=0.5ms, steps_per_action dt-steps]-->
  spikes on the readout neuron set (all descending+motor neurons, see
  scripts/diag_readout.py)
  --[leaky trace, tau=50ms, pooled into 64 pools: 48 named DN cell_type x
  side pools chosen by cross-channel response variance + 16 generic
  side-split pools, z-normalized with fixed screen stats]--> features (B, 64)
  --[decoder, trainable, anatomical-prior init]--> action = tanh(W_dec @ features + b_dec)  (B, 2)

Every parameter of the encoder+decoder lives in a single flat vector per
batch element: BrainPolicy is BATCHED, i.e. parameters are a (B, P) tensor
-- one full encoder+decoder per batch element (used by ES: one population
member per batch row), all sharing ONE FastBrain(batch=B) instance (the
brain weights themselves are never trained/mutated).

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

N_GROUPS = len(GROUP_NAMES)          # 12, one per FlyAddictionEnv obs channel
N_ENC_IN = N_GROUPS + 3              # 12 obs channels + 3 bilateral contrast features
N_POOLS = 64
N_ACTIONS = 2

# obs-channel indices of the three bilateral pairs, and which group index
# (== obs-channel index, since groups are 1:1 with obs channels) each
# contrast feature should additionally drive (+gain on the L group,
# -gain on the R group), sharpening lateralization from an untrained policy.
_BILATERAL_PAIRS = [(0, 1), (2, 3), (4, 5)]  # (food_L,food_R), (smoke_L,smoke_R), (reels_L,reels_R)
_CONTRAST_EPS = 1e-6
_CONTRAST_GAIN = 3.0
_DIAG_INIT = 4.0
_BIAS_INIT = -3.0

# v2: default anatomical-prior decoder init scale (see default_params below
# and results/screen/io_v2_choice.json "encoder_design_decision"). ES trains
# all weights from here; this only sets the untrained starting point.
DEFAULT_INIT_SCALE = 0.3

TAU_TRACE_MS = 50.0

# Task 2 (DAgger, flyrl.dagger_taxis) latency ablation: an OPTIONAL second,
# slower leaky trace over the same readout spikes, maintained alongside the
# tau=50ms trace above at negligible extra cost (one more elementwise
# multiply-add per sub-step). It is exposed (raw, unnormalized) via
# BrainPolicy.last_pooled_slow_raw purely for flyrl.dagger_taxis's own
# "does a slower trace help decodability" experiment -- it never feeds
# BrainPolicy.act()'s own decoder output (which is unchanged: still the
# single tau=50ms z-normalized 64-pool feature vector, exactly as v2/v3).
TAU_TRACE_SLOW_MS = 200.0

# Per-group encoder output ceiling in Hz (the "200" in v1's uniform
# `200.0 * sigmoid(...)`), now per-anatomical-group and decided empirically
# by the Task A screen (see flyrl.io_neurons.GROUP_MAX_RATE_HZ /
# results/screen/io_v2_choice.json) so each group's operating range avoids
# network ignition.
_GROUP_MAX_RATE_ARRAY = np.array([GROUP_MAX_RATE_HZ[name] for name in GROUP_NAMES], dtype=np.float32)

# Flat-parameter layout (all sizes fixed once N_GROUPS/N_POOLS are fixed).
_N_ENC_W = N_GROUPS * N_ENC_IN
_N_ENC_B = N_GROUPS
_N_DEC_W = N_ACTIONS * N_POOLS
_N_DEC_B = N_ACTIONS
N_PARAMS = _N_ENC_W + _N_ENC_B + _N_DEC_W + _N_DEC_B


def _load_readout_cache():
    if not READOUT_CACHE_PATH.exists():
        raise RuntimeError(
            f"{READOUT_CACHE_PATH} not found. Run `python scripts/diag_readout.py` "
            "first -- it decides and caches the readout neuron set (see spec: "
            "'FIRST run a diagnostic script')."
        )
    data = np.load(READOUT_CACHE_PATH)
    return (
        data["readout_idx"].astype(np.int64),
        data["pool_assign"].astype(np.int64),
        data["dan_idx"].astype(np.int64),
        int(data["n_pools"]),
        data["pool_mean"].astype(np.float32),
        data["pool_std"].astype(np.float32),
        data["pool_turn_sign"].astype(np.float32),
    )


def default_params(seed: int = 0, init_scale: float = DEFAULT_INIT_SCALE) -> np.ndarray:
    """The spec-prescribed initialization, flattened to a (N_PARAMS,) vector:
    encoder diagonal ~= +4 / bias ~= -3 (own-channel-driven groups), L/R
    groups additionally get +-contrast weight; decoder uses an
    ANATOMICAL-PRIOR init (v2, Task B): the turn output's weight on each of
    the 64 readout-pool features gets sign = pool_turn_sign (from the
    screen: +1 if that DN/motor pool responds more to LEFT-side v2 input
    channels, -1 if more to RIGHT), scaled by `init_scale`, so an untrained
    fly already turns TOWARD the stimulated side (empirically verified turn
    convention: action[0] > 0 turns the fly toward its LEFT, see the task
    report). Forward-speed output keeps a small-random weight + bias=+0.5
    (untrained fly already walks forward). ES trains every weight from here;
    `init_scale` only sets the starting point."""
    rng = np.random.default_rng(seed)

    W_enc = np.zeros((N_GROUPS, N_ENC_IN), dtype=np.float32)
    b_enc = np.full((N_GROUPS,), _BIAS_INIT, dtype=np.float32)
    for g in range(N_GROUPS):
        W_enc[g, g] = _DIAG_INIT
    for pair_i, (li, ri) in enumerate(_BILATERAL_PAIRS):
        contrast_col = N_GROUPS + pair_i
        W_enc[li, contrast_col] = _CONTRAST_GAIN
        W_enc[ri, contrast_col] = -_CONTRAST_GAIN

    W_dec = (rng.standard_normal((N_ACTIONS, N_POOLS)) * 0.05).astype(np.float32)
    try:
        _, _, _, _, _, _, pool_turn_sign = _load_readout_cache()
        W_dec[0, :] = init_scale * pool_turn_sign  # turn output: anatomical-prior sign
    except RuntimeError:
        pass  # readout cache not built yet (e.g. diag_readout.py hasn't run) -- fall back to random
    b_dec = np.array([0.0, 0.5], dtype=np.float32)

    theta = np.concatenate([
        W_enc.reshape(-1), b_enc.reshape(-1), W_dec.reshape(-1), b_dec.reshape(-1),
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
    """

    def __init__(self, batch: int, device: str = "cpu", dt: float = 0.5,
                 steps_per_action: int = 20, seed: int | None = None):
        self.batch = int(batch)
        self.device = torch.device(device)
        self.dt = float(dt)
        self.steps_per_action = int(steps_per_action)

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

        readout_idx, pool_assign, dan_idx, n_pools, pool_mean, pool_std, pool_turn_sign = _load_readout_cache()
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

        self.n_groups = N_GROUPS
        self.n_enc_in = N_ENC_IN
        self.n_pools = N_POOLS
        self.n_actions = N_ACTIONS
        self.n_params = N_PARAMS

        # Per-group encoder rate ceiling (Hz), decided empirically per
        # anatomical group by the Task A screen (see
        # flyrl.io_neurons.GROUP_MAX_RATE_HZ) instead of v1's single
        # constant, so each group's operating range avoids network ignition.
        self.group_max_rate = torch.as_tensor(_GROUP_MAX_RATE_ARRAY, dtype=torch.float32, device=self.device)

        # Leaky trace (tau=50ms) decay per dt sub-step.
        self.decay = float(np.exp(-self.dt / TAU_TRACE_MS))
        # Task 2: optional second, slower trace (tau=200ms) -- see
        # TAU_TRACE_SLOW_MS docstring above.
        self.decay_slow = float(np.exp(-self.dt / TAU_TRACE_SLOW_MS))

        self.theta = None
        self.W_enc = self.b_enc = self.W_dec = self.b_dec = None
        self.set_params(np.tile(default_params(), (self.batch, 1)))
        self.reset()

    # ------------------------------------------------------------------
    def set_params(self, theta):
        """theta: (B, N_PARAMS) array-like (numpy or torch). Parameters of
        batch row i only ever affect BrainPolicy.act's output for batch
        row i (no cross-batch mixing anywhere in encoder/decoder)."""
        theta = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
        assert theta.shape == (self.batch, self.n_params), (
            f"expected theta shape ({self.batch}, {self.n_params}), got {tuple(theta.shape)}"
        )
        self.theta = theta
        off = 0
        self.W_enc = theta[:, off:off + _N_ENC_W].reshape(self.batch, N_GROUPS, N_ENC_IN)
        off += _N_ENC_W
        self.b_enc = theta[:, off:off + _N_ENC_B].reshape(self.batch, N_GROUPS)
        off += _N_ENC_B
        self.W_dec = theta[:, off:off + _N_DEC_W].reshape(self.batch, N_ACTIONS, N_POOLS)
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
        self.last_pooled = torch.zeros((self.batch, N_POOLS), dtype=torch.float32, device=self.device)
        self.last_pooled_raw = torch.zeros((self.batch, N_POOLS), dtype=torch.float32, device=self.device)
        self.last_pooled_slow_raw = torch.zeros((self.batch, N_POOLS), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    def _encode(self, obs_t: torch.Tensor) -> torch.Tensor:
        """obs_t: (B, 12) -> group_rates_Hz (B, 12). Pure function of obs and
        the current encoder params; does not touch brain state."""
        contrasts = []
        for li, ri in _BILATERAL_PAIRS:
            L, R = obs_t[:, li], obs_t[:, ri]
            contrasts.append((L - R) / (L + R + _CONTRAST_EPS))
        obs_aug = torch.cat([obs_t] + [c.unsqueeze(1) for c in contrasts], dim=1)  # (B, 15)
        pre_enc = torch.einsum("bgf,bf->bg", self.W_enc, obs_aug) + self.b_enc  # (B, 12)
        return self.group_max_rate.unsqueeze(0) * torch.sigmoid(pre_enc)  # (B, 12) Hz, per-group ceiling

    @torch.no_grad()
    def group_rates(self, obs) -> np.ndarray:
        """Encoder output group_rates_Hz (B, n_groups) for `obs`, WITHOUT
        stepping the brain (diagnostics/tests only)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        assert obs_t.shape == (self.batch, N_GROUPS)
        return self._encode(obs_t).cpu().numpy()

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        """obs: (B, 12) array-like in [0,1] (FlyAddictionEnv.obs_channels
        order). Returns actions (B, 2) numpy array in [-1, 1]."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        assert obs_t.shape == (self.batch, N_GROUPS), (
            f"expected obs shape ({self.batch}, {N_GROUPS}), got {tuple(obs_t.shape)}"
        )

        group_rates = self._encode(obs_t)  # (B, 12) Hz
        neuron_rates = group_rates[:, self._group_id_per_neuron]  # (B, K_in)

        dan_spike_sum = torch.zeros((self.batch,), dtype=torch.float32, device=self.device)
        for _ in range(self.steps_per_action):
            spike = self.fb.step(rates=neuron_rates)  # (B, N) bool
            spike_f = spike.to(torch.float32)
            readout_spikes = spike_f.index_select(1, self.readout_idx)  # (B, n_readout)
            self.trace = self.trace * self.decay + readout_spikes
            self.trace_slow = self.trace_slow * self.decay_slow + readout_spikes  # Task 2 ablation only
            dan_spike_sum += spike_f.index_select(1, self.dan_idx).sum(dim=1)

        window_s = self.steps_per_action * self.dt / 1000.0
        self.last_dan_rate_hz = dan_spike_sum / max(self.n_dan, 1) / window_s

        pooled = torch.zeros((self.batch, N_POOLS), dtype=torch.float32, device=self.device)
        pooled.index_add_(1, self.pool_assign, self.trace)  # (B, 64) raw per-pool trace sum
        # v2: fixed z-normalization from the screen (spec), replacing v1's
        # /pool_counts/trace_norm_const heuristic normalization.
        pooled_normed = (pooled - self.pool_mean.unsqueeze(0)) / self.pool_std.unsqueeze(0)

        # Task 2 (flyrl.dagger_taxis): expose the raw and z-normed tau=50ms
        # pooled features (the decoder's actual input) plus the raw tau=200ms
        # pooled features, purely for the DAgger ridge-regression pipeline's
        # own feature extraction / latency ablation. None of this feeds back
        # into this method's own action output below.
        self.last_pooled = pooled_normed
        self.last_pooled_raw = pooled
        pooled_slow_raw = torch.zeros((self.batch, N_POOLS), dtype=torch.float32, device=self.device)
        pooled_slow_raw.index_add_(1, self.pool_assign, self.trace_slow)
        self.last_pooled_slow_raw = pooled_slow_raw

        pre_dec = torch.einsum("bof,bf->bo", self.W_dec, pooled_normed) + self.b_dec  # (B, 2)
        action = torch.tanh(pre_dec)
        return action.cpu().numpy()

    def features(self) -> np.ndarray:
        """(B, 64) z-normalized pooled readout features from the most recent
        act() call -- i.e. exactly what BrainPolicy.act() feeds its own
        decoder. Used by flyrl.dagger_taxis to fit an external ridge-
        regression decoder onto the same features BrainPolicy.set_params'
        decoder block would consume (Task 2)."""
        return self.last_pooled.cpu().numpy()

    def features_slow_raw(self) -> np.ndarray:
        """(B, 64) RAW (not z-normalized -- no screen stats exist for this
        trace) tau=200ms pooled features from the most recent act() call.
        Task 2 latency-ablation use only."""
        return self.last_pooled_slow_raw.cpu().numpy()

    def mean_dan_rate_hz(self) -> np.ndarray:
        """Per-batch-row mean DAN firing rate (Hz), computed over the most
        recent act() call's steps_per_action-step window (logging only --
        DAN activity never feeds the decoder)."""
        return self.last_dan_rate_hz.cpu().numpy()
