"""Task D: untrained-policy check -- does the anatomical-prior decoder
initialization (flyrl.policy.default_params, Task B) actually let the frozen
connectome pathway steer the fly, before any ES training?

For each modality (food, smoke, reels), over 16 diverse fixed seeds
(different per-seed source layout + fly start pose, NOT common-random-number
CRN -- this is an evaluation benchmark, not a training generation), runs a
120-step taxis sub-episode (flyrl.taxis.mask_obs, same masking as --mode
taxis) for two conditions:

  (a) prior-init: flyrl.policy.default_params() as-is (anatomical-prior
      decoder sign + encoder diagonal init).
  (b) zero-decoder: same encoder (so the brain still receives the same
      obs-driven sensory rates), but decoder weights zeroed except the
      forward bias (+0.5) -- i.e. the fly always walks straight in its
      initial random heading, completely ignoring brain activity. This is
      the "is the connectome pathway contributing anything at all" control.

Reports, per modality and condition: reach rate (fraction of the 16 seeds
that got within r_consume at least once in 120 steps) and mean distance
reduction (initial - final distance to target).

Writes results/screen/prior_init_check.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flyrl.policy import BrainPolicy, default_params, N_GROUPS, N_ENC_IN, N_ACTIONS, N_POOLS  # noqa: E402
from flyrl.addiction_env import VecFlyAddictionEnv  # noqa: E402
from flyrl.taxis import mask_obs, MODALITIES, _distances  # noqa: E402

N_SEEDS = 16
N_SUBSTEPS = 120
SEED_BASE = 5000
OUT_PATH = _REPO_ROOT / "results" / "screen" / "prior_init_check.json"


def zero_decoder_theta(batch: int) -> np.ndarray:
    """Same encoder as default_params (so the brain still gets driven), but
    decoder weights zeroed except the forward bias (+0.5) -- action is a
    CONSTANT [tanh(0)=0 turn, tanh(0.5)~=0.462 forward] regardless of brain
    activity: pure "walk straight in the initial heading" baseline."""
    theta = default_params(seed=0).copy()
    off = N_GROUPS * N_ENC_IN + N_GROUPS
    theta[off:off + N_ACTIONS * N_POOLS] = 0.0
    theta[off + N_ACTIONS * N_POOLS:] = np.array([0.0, 0.5], dtype=np.float32)
    return np.tile(theta, (batch, 1)).astype(np.float32)


def run_condition(theta_batch: np.ndarray, modality: str, seed_base: int):
    B = theta_batch.shape[0]
    policy = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=20, seed=0)
    vec_env = VecFlyAddictionEnv(num_envs=B, n_steps=300)
    policy.set_params(theta_batch)
    policy.reset()
    obs = vec_env.reset(seed=seed_base)  # per-index-offset seeds -> 16 diverse fixed episodes

    init_dist = _distances(vec_env, modality)
    r_consume = vec_env.envs[0].params.r_consume
    reached = np.zeros(B, dtype=bool)

    obs = mask_obs(obs, modality)
    for _ in range(N_SUBSTEPS):
        actions = policy.act(obs)
        obs, _rewards, _dones, _infos = vec_env.step(actions)
        obs = mask_obs(obs, modality)
        dist = _distances(vec_env, modality)
        reached |= dist <= r_consume

    final_dist = _distances(vec_env, modality)
    reduction = init_dist - final_dist
    return reached, reduction


def main():
    results = {}
    for modality in MODALITIES:
        theta_prior = np.tile(default_params(seed=0), (N_SEEDS, 1)).astype(np.float32)
        reached_p, red_p = run_condition(theta_prior, modality, seed_base=SEED_BASE)

        theta_zero = zero_decoder_theta(N_SEEDS)
        reached_z, red_z = run_condition(theta_zero, modality, seed_base=SEED_BASE)

        results[modality] = {
            "prior_init": {
                "reach_rate": float(reached_p.mean()),
                "mean_dist_reduction": float(red_p.mean()),
                "per_seed_dist_reduction": red_p.tolist(),
            },
            "zero_decoder": {
                "reach_rate": float(reached_z.mean()),
                "mean_dist_reduction": float(red_z.mean()),
                "per_seed_dist_reduction": red_z.tolist(),
            },
        }
        print(f"{modality:>6s}  prior_init: reach={reached_p.mean():.3f} "
              f"dist_reduction={red_p.mean():+.4f}   "
              f"zero_decoder: reach={reached_z.mean():.3f} dist_reduction={red_z.mean():+.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"n_seeds": N_SEEDS, "n_substeps": N_SUBSTEPS, "seed_base": SEED_BASE,
                   "results": results}, f, indent=2)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
