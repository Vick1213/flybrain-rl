"""Task 3 quick signal check (run BEFORE the long ES training run -- this is
the key risk gate, per spec: "We need to see that the parameterisation can
express both an addicted and a sober fly").

Starting from the v4.1 DAgger checkpoint (results/taxis_dagger_v41/ckpt.npz,
intensity-weighted contrast -- see flyrl.policy module docstring), evaluates
3 HAND-SET variants of the 19-param v4.1 steering encoder on 16 fixed seeds
x 300 steps each, in the FULL addiction env (all three sources present, no
obs masking), with gamma=1.0 (was 0.98 in the v4 signal check -- see
flyrl.train_es's --gamma docstring for why a myopic discount close to the
walk-to-source horizon was diagnosed as confounding fitness with
start-position luck):

  (i)   init      -- the DAgger checkpoint's own encoder, unmodified (equal
                      a=1.5/c=+4 attraction to all 3 sources)
  (ii)  smoker     -- c_smoke=+8, c_food=c_reels=0, a_food=a_reels=0: an
                      "addicted" parameterization with strong extra
                      attraction to smoke only and food/reels' own-side
                      intensity response zeroed too (not just their contrast)
  (iii) forager    -- c_food=+8, m_food_hunger=+3, c_smoke=c_reels=-3: a
                      "sober" parameterization: strong attraction to food
                      (amplified by hunger), AVERSION to smoke/reels

All other steering params (a other than ii's a_food/a_reels override, the
other m entries, h, b) and the entire decoder are left exactly as the
checkpoint. For each variant, reports true welfare (plain summed env
reward) and hijacked fitness (beta_nic=1.0, beta_jackpot=1.5, gamma=1.0 --
flyrl.train_es's --mode hijacked defaults), computed in ONE rollout per
variant (flyrl.train_es.run_episode_batch with mode="hijacked" always also
computes true welfare), plus source-time fractions and addiction metrics.

Success criterion (spec): (ii)'s hijacked fitness must beat BOTH (iii) and
(i), while (iii)'s true welfare must beat (ii)'s -- i.e. the two fitness
functions must rank (ii) and (iii) OPPOSITELY, showing the parameterization
can express both an addicted and a sober fly.

Usage: .venv/bin/python -m scripts.signal_check
Writes results/signal_check/report.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flyrl.policy import BrainPolicy, N_PARAMS, steering_param_names, unpack_steering_params  # noqa: E402
from flyrl.addiction_env import VecFlyAddictionEnv  # noqa: E402
from flyrl.train_es import run_episode_batch  # noqa: E402

CKPT_PATH = _REPO_ROOT / "results" / "taxis_dagger_v41" / "ckpt.npz"
OUT_DIR = _REPO_ROOT / "results" / "signal_check"
OUT_PATH = OUT_DIR / "report.json"

N_SEEDS = 16
N_STEPS = 300
EVAL_SEED = 555_000
BETA_NIC = 1.0
BETA_JACKPOT = 1.5
GAMMA = 1.0  # was 0.98 for the v4 signal check -- see flyrl.train_es --gamma docstring
DEFAULT_CONTRAST_WEIGHTING = "sqrt"  # fallback if the checkpoint's config lacks this key

METRIC_KEYS = ("frac_food", "frac_smoke", "frac_reels", "compulsion_smoke",
               "compulsion_reels", "mean_withdrawal", "final_tolerance")

VARIANTS = {
    "i_init_equal_attraction": {},
    "ii_addicted_smoke": {"c_food": 0.0, "c_smoke": 8.0, "c_reels": 0.0, "a_food": 0.0, "a_reels": 0.0},
    "iii_sober_food": {"c_food": 8.0, "m_food_hunger": 3.0, "c_smoke": -3.0, "c_reels": -3.0},
}


def make_variant_theta(base_theta: np.ndarray, overrides: dict) -> np.ndarray:
    theta = base_theta.copy()
    names = steering_param_names()
    idx = {n: i for i, n in enumerate(names)}
    for k, v in overrides.items():
        theta[idx[k]] = float(v)
    return theta


def main():
    assert CKPT_PATH.exists(), f"{CKPT_PATH} not found -- run flyrl.dagger_taxis --run-name taxis_dagger_v41 first"
    data = np.load(CKPT_PATH)
    base_theta = data["mean_theta"].astype(np.float64)
    assert base_theta.shape == (N_PARAMS,), f"ckpt mean_theta shape {base_theta.shape} != ({N_PARAMS},)"
    config = json.loads(str(data["config_json"])) if "config_json" in data else {}
    contrast_weighting = config.get("contrast_weighting", DEFAULT_CONTRAST_WEIGHTING)
    print(f"Using contrast_weighting={contrast_weighting!r} (from checkpoint config)")

    policy = BrainPolicy(batch=N_SEEDS, device="cpu", dt=0.5, steps_per_action=20, seed=0,
                          contrast_weighting=contrast_weighting)
    vec_env = VecFlyAddictionEnv(num_envs=N_SEEDS, n_steps=N_STEPS)

    report = {}
    for name, overrides in VARIANTS.items():
        theta = make_variant_theta(base_theta, overrides)
        theta_pop = np.tile(theta.astype(np.float32), (N_SEEDS, 1))
        result = run_episode_batch(policy, vec_env, theta_pop, seed=EVAL_SEED, n_steps=N_STEPS,
                                    mode="hijacked", beta_nic=BETA_NIC, beta_jackpot=BETA_JACKPOT,
                                    gamma=GAMMA, common_seed=False)
        msum = {k: float(np.mean([m[k] for m in result.metrics])) for k in METRIC_KEYS}
        row = {
            "overrides": overrides,
            "true_welfare_mean": float(result.welfare.mean()),
            "hijacked_fitness_mean": float(result.fitness.mean()),
            **msum,
            "steering_params": unpack_steering_params(theta),
        }
        report[name] = row
        print(f"{name:>28s}  welfare={row['true_welfare_mean']:8.3f}  "
              f"hijacked_fitness={row['hijacked_fitness_mean']:8.3f}  "
              f"food={row['frac_food']:.2f} smoke={row['frac_smoke']:.2f} reels={row['frac_reels']:.2f}")

    i_, ii, iii = report["i_init_equal_attraction"], report["ii_addicted_smoke"], report["iii_sober_food"]
    # Spec (Part E): hijacked fitness (ii) > (iii) AND (ii) > (i); true
    # welfare (iii) > (ii).
    ranking_ok = (
        ii["hijacked_fitness_mean"] > iii["hijacked_fitness_mean"]
        and ii["hijacked_fitness_mean"] > i_["hijacked_fitness_mean"]
        and iii["true_welfare_mean"] > ii["true_welfare_mean"]
    )
    report["_ranking_check"] = {
        "ii_hijacked_fitness_gt_iii": bool(ii["hijacked_fitness_mean"] > iii["hijacked_fitness_mean"]),
        "ii_hijacked_fitness_gt_i": bool(ii["hijacked_fitness_mean"] > i_["hijacked_fitness_mean"]),
        "iii_true_welfare_gt_ii": bool(iii["true_welfare_mean"] > ii["true_welfare_mean"]),
        "ranking_ok": bool(ranking_ok),
    }
    print(f"\nRanking check (spec success criterion): {'PASS' if ranking_ok else 'FAIL'}")
    print(json.dumps(report["_ranking_check"], indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {OUT_PATH}")
    return report


if __name__ == "__main__":
    main()
