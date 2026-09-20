"""Diagnostic + cache builder for BrainPolicy's v2 readout (Task B).

Run: .venv/bin/python -m scripts.diag_readout   (or `python scripts/diag_readout.py`)

v1 (kept for context) used a random, anatomically-blind assignment of all
descending+motor (DN/motor) neurons into 64 pools. v2 instead builds
features from the 12 chosen v2 anatomical input groups (flyrl.io_neurons,
decided by scripts/screen_entry_points.py -- see
results/screen/io_v2_choice.json):

  1. Stimulate each of the 12 v2 groups ALONE at its own operating rate
     (flyrl.io_neurons.GROUP_MAX_RATE_HZ) for 300 ms (dt=0.5ms, 600 steps),
     maintaining the SAME leaky trace (tau=50ms) BrainPolicy.act() uses,
     and read out the trace-pooled response of every (cell_type, side)
     DN/motor pool at the end of the window -- one sample per pool per
     channel (12 samples per pool).
  2. Rank (cell_type, side) pools by the VARIANCE of that response across
     the 12 channels (i.e. how much the pool discriminates between
     different sensory channels) and keep the top 48 as NAMED features.
  3. Every remaining DN/motor neuron (not in one of the 48 named pools)
     is assigned to one of 16 side-split generic pools (8 left + 8 right;
     unknown/center side neurons alternate between the two halves, as in
     v1), giving 48 + 16 = 64 total pools.
  4. z-normalization stats (mean, std) for all 64 pools are computed from
     the same 12 per-channel samples ("fixed stats from the screen").
  5. A per-pool turn-sign (+1/-1) is computed from whether the pool
     responds more when a LEFT-side v2 obs channel (food_odor_L,
     smoke_odor_L, reels_light_L) is driven vs. the matching RIGHT-side
     channel -- used by flyrl.policy.default_params for the anatomical-
     prior decoder initialization (turn TOWARD the stimulated side; see
     the empirical turn-sign-convention check in this task's report).

Outputs:
  - flyrl/readout_neurons.npz  -- readout_idx, pool_assign, dan_idx,
                                   n_pools, pool_mean, pool_std,
                                   pool_turn_sign, pool_names
  - results/diag_readout.csv          -- the v2 per-group x per-pool response table
  - results/diag_readout_decision.json -- summary of the v2 decision
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
N_NAMED_POOLS = 48
N_GENERIC_POOLS = 16
N_POOLS = N_NAMED_POOLS + N_GENERIC_POOLS  # 64
POOL_SEED = 0

# (left_channel, right_channel) pairs used to determine each pool's turn-sign
_LR_PAIRS = [("food_odor_L", "food_odor_R"), ("smoke_odor_L", "smoke_odor_R"),
             ("reels_light_L", "reels_light_R")]


def _stimulate_group_trace(fb, idx, rate_hz, dm_idx):
    """Drive `idx` alone at `rate_hz` for N_STEPS with a live leaky trace
    (tau=50ms) over dm_idx, matching BrainPolicy.act()'s bookkeeping.
    Returns the trace value (dm_idx,) at the end of the window."""
    decay = float(np.exp(-DT_MS / TAU_TRACE_MS))
    fb.reset()
    fb.set_input_neurons(idx)
    fb.set_exc_indices(idx)
    rates = torch.full((1, len(idx)), float(rate_hz), dtype=torch.float32)
    trace = torch.zeros((1, len(dm_idx)), dtype=torch.float32)
    dm_idx_t = torch.as_tensor(dm_idx, dtype=torch.long)
    for _ in range(N_STEPS):
        spike = fb.step(rates=rates)
        readout_spikes = spike.to(torch.float32).index_select(1, dm_idx_t)
        trace = trace * decay + readout_spikes
    return trace[0].numpy()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    fb = FastBrain(batch=1, device="cpu", dt=DT_MS, seed=0)
    groups = build_input_groups(fb)
    sizes = group_sizes(groups)
    print("v2 input group sizes:", sizes)

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

    # --- stimulate each of the 12 v2 groups alone, record trace @ end of window ---
    per_channel_trace = np.zeros((len(GROUP_NAMES), len(dm_idx)), dtype=np.float64)
    rows = []
    for gi, name in enumerate(GROUP_NAMES):
        idx = groups[name]
        rate = GROUP_MAX_RATE_HZ[name]
        t0 = time.time()
        trace = _stimulate_group_trace(fb, idx, rate, dm_idx)
        elapsed = time.time() - t0
        per_channel_trace[gi] = trace
        rows.append({
            "group": name, "n_input_neurons": len(idx), "rate_hz": rate,
            "trace_sum": float(trace.sum()), "n_dm_trace_nonzero": int((trace > 0).sum()),
            "sim_wall_s": elapsed,
        })
        print(f"  {name:>16s} @ {rate:6.1f}Hz: trace_sum={trace.sum():8.2f}  "
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

    # --- z-norm stats (mean, std over the 12 per-channel samples), for ALL 64 pools ---
    pooled_by_pool = np.zeros((len(GROUP_NAMES), N_POOLS), dtype=np.float64)
    for gi in range(len(GROUP_NAMES)):
        np.add.at(pooled_by_pool[gi], pool_assign, per_channel_trace[gi])
    pool_mean = pooled_by_pool.mean(axis=0)
    pool_std = pooled_by_pool.std(axis=0)
    pool_std = np.maximum(pool_std, 1e-3)  # floor to avoid div-by-zero for always-silent pools

    # --- turn-sign: does this pool respond more to LEFT-side v2 channels than RIGHT? ---
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
