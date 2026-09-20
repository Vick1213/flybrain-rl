"""Diagnostic + cache builder for BrainPolicy's readout (Task B, rebuilt for
I/O v4 -- Task 1).

Run: .venv/bin/python -m scripts.diag_readout   (or `python scripts/diag_readout.py`)

v1 (kept for context) used a random, anatomically-blind assignment of all
descending+motor (DN/motor) neurons into 64 pools. v2/v3 built features
from the (12) v2/v3 anatomical input groups. v4 (this version) reruns the
IDENTICAL procedure against v4's 8 anatomical input groups (flyrl.io_neurons:
the shared steer_L/steer_R pair + 6 unchanged contact/interoceptive
groups) -- no change to the algorithm itself, only to which/how-many
groups get stimulated, since flyrl.io_neurons.GROUP_NAMES now has 8
entries instead of 12:

  1. Stimulate each of the 8 v4 groups ALONE at its own operating rate
     (flyrl.io_neurons.GROUP_MAX_RATE_HZ) for 300 ms (dt=0.5ms, 600 steps),
     maintaining the SAME leaky trace (tau=50ms) BrainPolicy.act() uses,
     and read out the trace-pooled response of every (cell_type, side)
     DN/motor pool at the end of the window -- one sample per pool per
     channel (8 samples per pool). A SECOND, slower (tau=200ms) trace is
     maintained in parallel at negligible extra cost, for Task 1's optional
     slow-trace decoder feature (flyrl.policy.USE_SLOW_TRACE) -- see step 4.
  2. Rank (cell_type, side) pools by the VARIANCE of that response across
     the 8 channels (i.e. how much the pool discriminates between
     different sensory channels) and keep the top 48 as NAMED features.
  3. Every remaining DN/motor neuron (not in one of the 48 named pools)
     is assigned to one of 16 side-split generic pools (8 left + 8 right;
     unknown/center side neurons alternate between the two halves, as in
     v1), giving 48 + 16 = 64 total pools.
  4. z-normalization stats (mean, std) for all 64 pools are computed from
     the same 8 per-channel samples ("fixed stats from the screen"), for
     BOTH the tau=50ms trace (pool_mean/pool_std) and the tau=200ms trace
     (pool_mean_slow/pool_std_slow -- used only if flyrl.policy.
     USE_SLOW_TRACE is True).
  5. A per-pool turn-sign (+1/-1) is computed from whether the pool
     responds more when the LEFT steering group (steer_L) is driven vs.
     the RIGHT (steer_R) -- used by flyrl.policy.default_params for the
     anatomical-prior decoder initialization (turn TOWARD the stimulated
     side; see the empirical turn-sign-convention check in this task's
     report). v4 has only ONE lateralized input pair (steer_L/steer_R,
     shared by all three directional sources), so this replaces v2/v3's
     three-pair (food/smoke/reels) average.

Outputs:
  - flyrl/readout_neurons.npz  -- readout_idx, pool_assign, dan_idx,
                                   n_pools, pool_mean, pool_std,
                                   pool_mean_slow, pool_std_slow,
                                   pool_turn_sign, pool_names
  - results/diag_readout.csv          -- the v4 per-group x per-pool response table
  - results/diag_readout_decision.json -- summary of the v4 decision
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flyrl.fastbrain import FastBrain  # noqa: E402
from flyrl.io_neurons import (  # noqa: E402
    GROUP_NAMES, GROUP_MAX_RATE_HZ, build_input_groups, group_sizes,
    descending_motor_with_type, dan_indices, union_and_group_ids,
)

RESULTS_DIR = _REPO_ROOT / "results"
READOUT_CACHE = _REPO_ROOT / "flyrl" / "readout_neurons.npz"
DECISION_JSON = RESULTS_DIR / "diag_readout_decision.json"
DIAG_CSV = RESULTS_DIR / "diag_readout.csv"

WINDOW_MS = 300.0
DT_MS = 0.5
N_STEPS = int(round(WINDOW_MS / DT_MS))  # 600
TAU_TRACE_MS = 50.0
TAU_TRACE_SLOW_MS = 200.0
N_NAMED_POOLS = 48
N_GENERIC_POOLS = 16
N_POOLS = N_NAMED_POOLS + N_GENERIC_POOLS  # 64
POOL_SEED = 0

# (left_channel, right_channel) pair used to determine each pool's turn-sign.
# v4: only ONE lateralized input pair exists (steer_L/steer_R, shared by all
# three directional sources) -- replaces v2/v3's three-pair average.
_LR_PAIRS = [("steer_L", "steer_R")]


def _stimulate_group_trace(fb, idx, rate_hz, dm_idx):
    """Drive `idx` alone at `rate_hz` for N_STEPS with live leaky traces
    (tau=50ms AND tau=200ms) over dm_idx, matching BrainPolicy.act()'s
    bookkeeping. Returns (trace_fast, trace_slow), each (dm_idx,), at the
    end of the window."""
    decay = float(np.exp(-DT_MS / TAU_TRACE_MS))
    decay_slow = float(np.exp(-DT_MS / TAU_TRACE_SLOW_MS))
    fb.reset()
    fb.set_input_neurons(idx)
    fb.set_exc_indices(idx)
    rates = torch.full((1, len(idx)), float(rate_hz), dtype=torch.float32)
    trace = torch.zeros((1, len(dm_idx)), dtype=torch.float32)
    trace_slow = torch.zeros((1, len(dm_idx)), dtype=torch.float32)
    dm_idx_t = torch.as_tensor(dm_idx, dtype=torch.long)
    for _ in range(N_STEPS):
        spike = fb.step(rates=rates)
        readout_spikes = spike.to(torch.float32).index_select(1, dm_idx_t)
        trace = trace * decay + readout_spikes
        trace_slow = trace_slow * decay_slow + readout_spikes
    return trace[0].numpy(), trace_slow[0].numpy()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    fb = FastBrain(batch=1, device="cpu", dt=DT_MS, seed=0)
    groups = build_input_groups(fb)
    sizes = group_sizes(groups)
    print("v4 input group sizes:", sizes)

    dm_idx, dm_side, dm_ct = descending_motor_with_type(fb)
    dan_idx = dan_indices(fb)
    print(f"Descending+motor readout set: {len(dm_idx)} neurons "
          f"(L={int((dm_side=='left').sum())}, R={int((dm_side=='right').sum())}, "
          f"C={int((~np.isin(dm_side, ['left','right'])).sum())})")
    print(f"DAN candidates (logging only): {len(dan_idx)}")

    # --- (cell_type, side) pool keys for all dm neurons ---
    pool_keys = np.array([f"{ct}|{side}" for ct, side in zip(dm_ct, dm_side)])
    uniq_keys, key_inverse = np.unique(pool_keys, return_inverse=True)
    n_unique = len(uniq_keys)

    # --- stimulate each of the 8 v4 groups alone, record trace @ end of window ---
    per_channel_trace = np.zeros((len(GROUP_NAMES), len(dm_idx)), dtype=np.float64)
    per_channel_trace_slow = np.zeros((len(GROUP_NAMES), len(dm_idx)), dtype=np.float64)
    rows = []
    for gi, name in enumerate(GROUP_NAMES):
        idx = groups[name]
        rate = GROUP_MAX_RATE_HZ[name]
        t0 = time.time()
        trace, trace_slow = _stimulate_group_trace(fb, idx, rate, dm_idx)
        elapsed = time.time() - t0
        per_channel_trace[gi] = trace
        per_channel_trace_slow[gi] = trace_slow
        rows.append({
            "group": name, "n_input_neurons": len(idx), "rate_hz": rate,
            "trace_sum": float(trace.sum()), "n_dm_trace_nonzero": int((trace > 0).sum()),
            "trace_slow_sum": float(trace_slow.sum()),
            "sim_wall_s": elapsed,
        })
        print(f"  {name:>16s} @ {rate:6.1f}Hz: trace_sum={trace.sum():8.2f}  "
              f"trace_slow_sum={trace_slow.sum():8.2f}  "
              f"n_nonzero={int((trace>0).sum()):4d}  ({elapsed:.2f}s)")

    with open(DIAG_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {DIAG_CSV}")

    # --- pool the per-channel trace by (cell_type, side) key ---
    # pooled_by_key[channel, key] = sum of trace over dm neurons with that key
    pooled_by_key = np.zeros((len(GROUP_NAMES), n_unique), dtype=np.float64)
    for gi in range(len(GROUP_NAMES)):
        np.add.at(pooled_by_key[gi], key_inverse, per_channel_trace[gi])

    # --- rank keys by cross-channel variance, keep top 48 as named pools ---
    key_variance = pooled_by_key.var(axis=0)
    order = np.argsort(-key_variance)
    named_key_ids = order[:N_NAMED_POOLS]
    named_keys = uniq_keys[named_key_ids]
    print(f"Top {N_NAMED_POOLS} named (cell_type,side) pools by cross-channel variance "
          f"(showing 10): {named_keys[:10].tolist()}")

    # --- assign every dm neuron to a pool: named (0..47) or generic (48..63) ---
    pool_assign = np.full(len(dm_idx), -1, dtype=np.int64)
    named_key_set = {k: i for i, k in enumerate(named_keys.tolist())}
    for n_i, key in enumerate(pool_keys):
        if key in named_key_set:
            pool_assign[n_i] = named_key_set[key]

    remaining = np.nonzero(pool_assign < 0)[0]
    rem_side = dm_side[remaining]
    left_mask = rem_side == "left"
    right_mask = rem_side == "right"
    other_mask = ~(left_mask | right_mask)
    other_positions = np.nonzero(other_mask)[0]
    for k, pos in enumerate(other_positions):
        if k % 2 == 0:
            left_mask[pos] = True
        else:
            right_mask[pos] = True

    rng = np.random.default_rng(POOL_SEED)
    n_generic_half = N_GENERIC_POOLS // 2
    left_positions = remaining[np.nonzero(left_mask)[0]]
    right_positions = remaining[np.nonzero(right_mask)[0]]
    pool_assign[left_positions] = N_NAMED_POOLS + rng.integers(0, n_generic_half, size=len(left_positions))
    pool_assign[right_positions] = N_NAMED_POOLS + n_generic_half + rng.integers(0, n_generic_half, size=len(right_positions))
    assert np.all(pool_assign >= 0), "every readout neuron must get a pool"
    assert pool_assign.max() < N_POOLS

    pool_names = [f"{k.split('|')[0]}_{k.split('|')[1][0].upper()}" for k in named_keys.tolist()]
    for hemi in ("L", "R"):
        for k in range(n_generic_half):
            pool_names.append(f"generic_{hemi}{k}")
    assert len(pool_names) == N_POOLS

    # --- z-norm stats (mean, std over the 8 per-channel samples), for ALL 64 pools ---
    pooled_by_pool = np.zeros((len(GROUP_NAMES), N_POOLS), dtype=np.float64)
    pooled_by_pool_slow = np.zeros((len(GROUP_NAMES), N_POOLS), dtype=np.float64)
    for gi in range(len(GROUP_NAMES)):
        np.add.at(pooled_by_pool[gi], pool_assign, per_channel_trace[gi])
        np.add.at(pooled_by_pool_slow[gi], pool_assign, per_channel_trace_slow[gi])
    pool_mean = pooled_by_pool.mean(axis=0)
    pool_std = pooled_by_pool.std(axis=0)
    pool_std = np.maximum(pool_std, 1e-3)  # floor to avoid div-by-zero for always-silent pools
    # Task 1: fixed z-norm stats for the optional tau=200ms slow-trace decoder
    # feature (flyrl.policy.USE_SLOW_TRACE), built the same way from the same
    # per-channel stimulation samples.
    pool_mean_slow = pooled_by_pool_slow.mean(axis=0)
    pool_std_slow = pooled_by_pool_slow.std(axis=0)
    pool_std_slow = np.maximum(pool_std_slow, 1e-3)

    # --- turn-sign: does this pool respond more to LEFT-side (steer_L) than RIGHT (steer_R)? ---
    name_to_gi = {name: i for i, name in enumerate(GROUP_NAMES)}
    left_resp = np.zeros(N_POOLS, dtype=np.float64)
    right_resp = np.zeros(N_POOLS, dtype=np.float64)
    for left_name, right_name in _LR_PAIRS:
        left_resp += pooled_by_pool[name_to_gi[left_name]]
        right_resp += pooled_by_pool[name_to_gi[right_name]]
    pool_turn_sign = np.where(left_resp >= right_resp, 1.0, -1.0)
    # Pools with literally no signal on either side get a deterministic
    # alternating sign instead of an arbitrary "left wins" default, so the
    # anatomical prior doesn't silently bias every dead pool the same way.
    tied = (left_resp == 0) & (right_resp == 0)
    tied_idx = np.nonzero(tied)[0]
    pool_turn_sign[tied_idx[0::2]] = 1.0
    pool_turn_sign[tied_idx[1::2]] = -1.0

    np.savez(
        READOUT_CACHE,
        readout_idx=dm_idx, pool_assign=pool_assign, dan_idx=dan_idx,
        n_pools=np.array(N_POOLS), pool_mean=pool_mean, pool_std=pool_std,
        pool_mean_slow=pool_mean_slow, pool_std_slow=pool_std_slow,
        pool_turn_sign=pool_turn_sign, pool_names=np.array(pool_names),
    )
    print(f"Saved readout cache to {READOUT_CACHE} "
          f"(readout_idx: {len(dm_idx)}, pools: {N_POOLS} = {N_NAMED_POOLS} named + {N_GENERIC_POOLS} generic)")

    decision = {
        "window_ms": WINDOW_MS, "dt_ms": DT_MS, "tau_trace_ms": TAU_TRACE_MS,
        "n_dm_readout": int(len(dm_idx)), "n_dan_candidates": int(len(dan_idx)),
        "n_named_pools": N_NAMED_POOLS, "n_generic_pools": N_GENERIC_POOLS, "n_pools": N_POOLS,
        "named_pool_names": pool_names[:N_NAMED_POOLS],
        "group_max_rate_hz": GROUP_MAX_RATE_HZ,
        "input_group_sizes": sizes,
        "pool_turn_sign_left_count": int((pool_turn_sign > 0).sum()),
        "pool_turn_sign_right_count": int((pool_turn_sign < 0).sum()),
        "wall_s_total": time.time() - t_start,
    }
    with open(DECISION_JSON, "w") as f:
        json.dump(decision, f, indent=2)
    print(f"Saved decision summary to {DECISION_JSON}")
    print(f"Total diagnostic wall time: {decision['wall_s_total']:.1f}s")


if __name__ == "__main__":
    main()
