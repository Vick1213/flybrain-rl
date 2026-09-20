"""OpenAI-style Evolution Strategies trainer for BrainPolicy on
FlyAddictionEnv, driven by a FROZEN fly-connectome spiking brain.

CLI
---
    python -m flyrl.train_es --mode hijacked --gens 150 --run-name addicted --device cpu

Algorithm
---------
Population B=32 = 16 antithetic direction pairs (theta = mean +- sigma*eps).
All 32 population members share ONE FastBrain(batch=32) (via one BrainPolicy
instance) and, within a generation, the SAME environment seed ("common
random numbers": fitness differences across the population are then
attributable to parameter differences, not environment luck). A new
environment seed is drawn each generation. Fitness is rank-normalized
(centered ranks in [-0.5, 0.5]) before estimating the ES gradient, which is
fed through Adam to update the single MEAN parameter vector (sigma is held
constant by default; --sigma-decay optionally multiplies it each gen).

Two fitness modes (--mode):
  welfare  : the "sober fly" -- fitness = discounted... no: PLAIN sum of
             true env reward over the episode (the control condition).
  hijacked : the "addicted fly" -- models drugs hijacking the learning
             signal (Redish 2004-style non-compensable dopamine). Per-step
             learning signal = true_reward + beta_nic*[at=='smoke'] (a
             constant bonus that does NOT habituate with tolerance) +
             beta_jackpot*[jackpot], summed with a myopic discount gamma^t
             over the episode. True welfare (plain summed true reward) is
             ALWAYS additionally computed and logged, in both modes, so the
             welfare/fitness gap the hijack creates is visible.

NOTE on "common random numbers": VecFlyAddictionEnv.reset(seed) (as given)
assigns each sub-env i a DIFFERENT seed (seed+i) by design. To get the
common-random-numbers property required by the spec (all population members
see the same env realization within a generation), this module drives each
FlyAddictionEnv sub-env's `.reset(seed=...)` directly with the SAME seed for
every sub-env (see `_vec_reset_common_seed`), instead of calling
VecFlyAddictionEnv.reset(). VecFlyAddictionEnv.step() is used unmodified.
Evaluation (`_evaluate_mean`) uses the standard per-index-offset reset
(8 distinct fixed seeds), since eval wants a diverse fixed benchmark, not
CRN across ES noise directions.

Task 3 (I/O v4 preference learning): the 19 v4 steering-encoder params
(a/c/m/h/b -- see flyrl.policy) get a LARGER, per-parameter ES mutation
sigma (--sigma-steer, spec 0.3) than the decoder + 12 contact/intero
encoder params (--sigma, spec 0.05) -- see
flyrl.policy.steering_param_sigma_vector and build_population/es_gradient's
docstrings, which both accept a (P,) sigma array as well as a scalar. Every
--eval-every (spec: 5) generations the noiseless mean is evaluated on
--eval-episodes (spec: 8) fixed seeds and eval.csv logs BOTH true welfare
and the mode-dependent fitness (so the welfare/fitness gap the hijack
creates is visible over training, not just at the end), addiction metrics,
DAN rates, AND the current value of every steering param (a, c, m, h, b),
so preference drift toward/away from any source is visible directly in
eval.csv.

Task 4 (per-parameter Adam lr): per-parameter ES sigma alone is not enough
-- Adam moves each parameter by at most ~lr per generation regardless of
sigma, so the O(1)-scale steering params (sigma=0.3) still crawl under a
single small shared lr. --lr-steer (spec: 0.2) gives the same 19 steering
params a LARGER Adam lr than --lr (spec: 0.03) gives the rest, built into a
(P,) vector the same way as sigma_vec (see Adam and
flyrl.policy.steering_param_sigma_vector). After each update, the steering
params are clipped to a sane box (clip_steering_params: a/c/m/h in
[-8, 8], b_steer in [-6, 2]) so the larger, faster steps cannot run the
encoder's sigmoid drives away.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from flyrl.addiction_env import VecFlyAddictionEnv, addiction_metrics
from flyrl.policy import (
    BrainPolicy, default_params, N_PARAMS, N_STEER_PARAMS,
    steering_param_names, unpack_steering_params, steering_param_sigma_vector,
)
from flyrl.taxis import MODALITIES, run_taxis_generation

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

METRIC_KEYS = ("frac_food", "frac_smoke", "frac_reels", "compulsion_smoke",
               "compulsion_reels", "mean_withdrawal", "final_tolerance")


# ---------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------

def _vec_reset_common_seed(vec_env: VecFlyAddictionEnv, seed: int) -> np.ndarray:
    """Reset every sub-env of `vec_env` with the IDENTICAL seed (common
    random numbers across the ES population for this generation), instead
    of VecFlyAddictionEnv.reset()'s per-index seed+i offset."""
    obs_list = []
    for env in vec_env.envs:
        obs, _info = env.reset(seed=seed)
        obs_list.append(obs)
    vec_env._last_obs = obs_list
    vec_env._done[:] = False
    return np.stack(obs_list, axis=0)


class EpisodeResult:
    __slots__ = ("fitness", "welfare", "metrics", "mean_dan_rate", "mean_dan_rate_driven")

    def __init__(self, fitness, welfare, metrics, mean_dan_rate, mean_dan_rate_driven):
        self.fitness = fitness            # (B,) fitness used for ES (mode-dependent)
        self.welfare = welfare            # (B,) true summed env reward, always
        self.metrics = metrics            # list of B addiction_metrics dicts
        self.mean_dan_rate = mean_dan_rate  # (B,) Hz, non-input ("other") DANs only, averaged over episode
        self.mean_dan_rate_driven = mean_dan_rate_driven  # (B,) Hz, directly-driven (PAM+PPL input) DANs only


def run_episode_batch(policy: BrainPolicy, vec_env: VecFlyAddictionEnv, theta_pop: np.ndarray,
                       seed: int, n_steps: int, mode: str,
                       beta_nic: float = 1.0, beta_jackpot: float = 1.5, gamma: float = 0.98,
                       common_seed: bool = True) -> EpisodeResult:
    """Roll out `n_steps` of FlyAddictionEnv for a population of B policies
    (theta_pop: (B, N_PARAMS)), sharing one BrainPolicy(batch=B). Returns an
    EpisodeResult with both the mode-dependent ES fitness and the always-
    computed true welfare."""
    B = theta_pop.shape[0]
    assert vec_env.num_envs == B and policy.batch == B

    policy.set_params(theta_pop)
    policy.reset()
    if common_seed:
        obs = _vec_reset_common_seed(vec_env, seed)
    else:
        obs = vec_env.reset(seed=seed)

    welfare = np.zeros(B, dtype=np.float64)
    fitness = np.zeros(B, dtype=np.float64)
    dan_rate_sum = np.zeros(B, dtype=np.float64)
    dan_rate_driven_sum = np.zeros(B, dtype=np.float64)
    discount = 1.0
    episode_infos = [[] for _ in range(B)]

    for _ in range(n_steps):
        actions = policy.act(obs)
        obs, rewards, dones, infos = vec_env.step(actions)
        rewards = rewards.astype(np.float64)
        welfare += rewards
        if mode == "hijacked":
            bonus = np.zeros(B, dtype=np.float64)
            for i, info in enumerate(infos):
                if not info:
                    continue
                if info.get("at") == "smoke":
                    bonus[i] += beta_nic
                if info.get("jackpot"):
                    bonus[i] += beta_jackpot
            fitness += discount * (rewards + bonus)
            discount *= gamma
        for i, info in enumerate(infos):
            if info:
                episode_infos[i].append(info)
        dan_rate_sum += policy.mean_dan_rate_hz()
        dan_rate_driven_sum += policy.dan_rate_driven_hz()

    if mode == "welfare":
        fitness = welfare.copy()

    metrics = [addiction_metrics(episode_infos[i]) for i in range(B)]
    mean_dan_rate = dan_rate_sum / n_steps
    mean_dan_rate_driven = dan_rate_driven_sum / n_steps
    return EpisodeResult(fitness=fitness, welfare=welfare, metrics=metrics,
                          mean_dan_rate=mean_dan_rate, mean_dan_rate_driven=mean_dan_rate_driven)


# ---------------------------------------------------------------------
# ES machinery
# ---------------------------------------------------------------------

def rank_transform(fitness: np.ndarray) -> np.ndarray:
    """Centered ranks in [-0.5, 0.5], best fitness -> +0.5."""
    order = np.argsort(fitness)
    ranks = np.empty(len(fitness), dtype=np.float64)
    ranks[order] = np.arange(len(fitness), dtype=np.float64)
    return ranks / (len(fitness) - 1) - 0.5


def es_gradient(eps_half: np.ndarray, fitness: np.ndarray, sigma) -> np.ndarray:
    """Antithetic OpenAI-ES gradient estimate. eps_half: (H, P) sampled
    directions; fitness: (2H,) for population [mean+sigma*eps, mean-sigma*eps].
    `sigma` may be a scalar or a (P,) per-parameter array (Task 3: the 19
    steering-encoder params get a larger sigma than the rest -- see
    flyrl.policy.steering_param_sigma_vector) -- the division below is then
    elementwise, giving each parameter i a gradient estimate scaled by its
    OWN sigma_i, exactly the per-parameter-sigma OpenAI-ES estimator."""
    H = eps_half.shape[0]
    eps_full = np.concatenate([eps_half, -eps_half], axis=0)  # (2H, P)
    centered = rank_transform(fitness)
    grad = (eps_full * centered[:, None]).sum(axis=0) / (2 * H * sigma)
    return grad


class Adam:
    """Adam optimizer over a flat (P,) parameter vector. `lr` may be a
    python/numpy scalar (every parameter shares one lr) or a (P,) per-
    parameter array (Task 4: --lr-steer gives the 19 O(1)-scale steering
    params -- see flyrl.policy.steering_param_names -- a LARGER lr than the
    rest, built the same way as the per-parameter ES sigma; see
    steering_param_sigma_vector). mhat/vhat in step() are always (P,)
    arrays, so `self.lr * mhat / ...` broadcasts correctly whether self.lr
    is a scalar or a (P,) array -- no other logic changes needed for
    per-parameter lr."""

    def __init__(self, n_params, lr=0.03, beta1=0.9, beta2=0.999, eps=1e-8):
        lr_arr = np.asarray(lr, dtype=np.float64)
        if lr_arr.ndim > 0:
            assert lr_arr.shape == (n_params,), (
                f"lr array must have shape ({n_params},), got {lr_arr.shape}"
            )
            self.lr = lr_arr
        else:
            self.lr = float(lr_arr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(n_params, dtype=np.float64)
        self.v = np.zeros(n_params, dtype=np.float64)
        self.t = 0

    def step(self, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        mhat = self.m / (1 - self.beta1 ** self.t)
        vhat = self.v / (1 - self.beta2 ** self.t)
        return self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def state_dict(self):
        return {"m": self.m, "v": self.v, "t": np.array(self.t)}

    def load_state_dict(self, d):
        self.m = np.array(d["m"], dtype=np.float64)
        self.v = np.array(d["v"], dtype=np.float64)
        self.t = int(d["t"])


def build_population(mean_theta: np.ndarray, sigma, half: int, rng: np.random.Generator):
    """`sigma` may be a scalar or a (P,) per-parameter array -- see
    es_gradient's docstring (Task 3 per-parameter sigma scaling)."""
    eps_half = rng.standard_normal((half, mean_theta.shape[0])).astype(np.float64)
    theta_pop = np.concatenate([
        mean_theta[None, :] + sigma * eps_half,
        mean_theta[None, :] - sigma * eps_half,
    ], axis=0)
    return theta_pop.astype(np.float32), eps_half


# Task 4 (per-parameter Adam lr): now that --lr-steer lets the 19
# steering-encoder params move much faster per generation than the rest,
# clip them to a sane box after every update so the sigmoid drives (see
# flyrl.policy.BrainPolicy._encode) cannot run away. Indices found via the
# existing steering_param_names() helper (fixed layout: b_steer, then
# a/c/m/h -- see its docstring).
_STEER_NAMES = steering_param_names()
_B_STEER_IDX = np.array([i for i, n in enumerate(_STEER_NAMES) if n == "b_steer"])
_OTHER_STEER_IDX = np.array([i for i, n in enumerate(_STEER_NAMES) if n != "b_steer"])
STEER_CLIP_B = (-6.0, 2.0)      # b_steer box
STEER_CLIP_ACMH = (-8.0, 8.0)   # a, c, m, h box


def clip_steering_params(theta: np.ndarray) -> np.ndarray:
    """Clip the 19 steering-encoder params (see steering_param_names) to a
    sane box: b_steer in STEER_CLIP_B, a/c/m/h in STEER_CLIP_ACMH. All
    other (non-steering) params are returned unchanged. Returns a new
    array (does not mutate `theta` in place)."""
    theta = theta.copy()
    theta[_B_STEER_IDX] = np.clip(theta[_B_STEER_IDX], *STEER_CLIP_B)
    theta[_OTHER_STEER_IDX] = np.clip(theta[_OTHER_STEER_IDX], *STEER_CLIP_ACMH)
    return theta


# ---------------------------------------------------------------------
# Logging / checkpointing
# ---------------------------------------------------------------------

LOG_COLUMNS = [
    "gen", "wall_s", "fitness_mean", "fitness_max", "welfare_mean", "welfare_max",
    "frac_food", "frac_smoke", "frac_reels", "compulsion_smoke", "compulsion_reels",
    "mean_withdrawal", "final_tolerance", "mean_dan_rate_hz", "mean_dan_rate_driven_hz",
]

# Task 3: every --eval-every generations, evaluate the NOISELESS mean on
# --eval-episodes fixed seeds and log both true welfare and the
# mode-dependent (hijacked or welfare) fitness, addiction metrics, DAN
# rates, AND the current values of the 19 steering-encoder params (a, c, m,
# h, b) -- so preference drift toward/away from any source is visible
# directly in eval.csv over the course of training.
EVAL_COLUMNS = [
    "gen", "welfare_mean", "welfare_max", "fitness_mean", "fitness_max",
    "frac_food", "frac_smoke", "frac_reels", "compulsion_smoke", "compulsion_reels",
    "mean_withdrawal", "final_tolerance", "mean_dan_rate_hz", "mean_dan_rate_driven_hz",
] + steering_param_names()

# Task C: --mode taxis logging (per-modality reach rate + mean distance
# reduction, in addition to the pooled fitness ES actually optimizes).
TAXIS_LOG_COLUMNS = [
    "gen", "wall_s", "fitness_mean", "fitness_max",
    "reach_food", "reach_smoke", "reach_reels",
    "dist_reduction_food", "dist_reduction_smoke", "dist_reduction_reels",
]
TAXIS_EVAL_COLUMNS = [
    "gen", "fitness_mean", "fitness_max",
    "reach_food", "reach_smoke", "reach_reels",
    "dist_reduction_food", "dist_reduction_smoke", "dist_reduction_reels",
]


def _summarize_taxis(result) -> dict:
    out = {}
    for modality in MODALITIES:
        out[f"reach_{modality}"] = float(np.mean(result.reach_rate[modality]))
        out[f"dist_reduction_{modality}"] = float(np.mean(result.dist_reduction[modality]))
    return out


def _append_csv_row(path: Path, columns, row: dict):
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _summarize_metrics(metrics_list):
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in METRIC_KEYS}


def save_checkpoint(run_dir: Path, gen: int, mean_theta: np.ndarray, adam: Adam, config: dict):
    ckpt_path = run_dir / "ckpt.npz"
    adam_state = adam.state_dict()
    np.savez(
        ckpt_path,
        gen=np.array(gen),
        mean_theta=mean_theta.astype(np.float32),
        adam_m=adam_state["m"], adam_v=adam_state["v"], adam_t=adam_state["t"],
        config_json=np.array(json.dumps(config)),
    )


def load_checkpoint(run_dir: Path):
    ckpt_path = run_dir / "ckpt.npz"
    data = np.load(ckpt_path, allow_pickle=False)
    config = json.loads(str(data["config_json"]))
    mean_theta = data["mean_theta"].astype(np.float64)
    # Task 4: rebuild the same (P,) per-parameter lr vector used during
    # training (config["lr_steer"] falls back to config["lr"] for
    # checkpoints saved before --lr-steer existed, i.e. a uniform lr).
    # Adam's own state (m, v, t) is unchanged in shape by this -- only the
    # lr multiplier used going forward changes.
    lr_steer = config.get("lr_steer", config["lr"])
    lr_vec = steering_param_sigma_vector(config["lr"], lr_steer, mean_theta.shape[0])
    adam = Adam(mean_theta.shape[0], lr=lr_vec)
    adam.load_state_dict({"m": data["adam_m"], "v": data["adam_v"], "t": data["adam_t"]})
    gen = int(data["gen"])
    return gen, mean_theta, adam, config


# ---------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------

def _evaluate_mean(eval_policy: BrainPolicy, eval_env: VecFlyAddictionEnv, mean_theta: np.ndarray,
                    n_steps: int, mode: str, beta_nic: float, beta_jackpot: float, gamma: float,
                    eval_seed: int):
    """Evaluate the noiseless mean parameters on `eval_policy.batch` fixed
    seeds (standard per-index-offset reset -> a diverse, fixed benchmark)."""
    theta = np.tile(mean_theta.astype(np.float32), (eval_policy.batch, 1))
    result = run_episode_batch(eval_policy, eval_env, theta, seed=eval_seed, n_steps=n_steps,
                                mode=mode, beta_nic=beta_nic, beta_jackpot=beta_jackpot, gamma=gamma,
                                common_seed=False)
    return result


def _evaluate_mean_taxis(eval_policy: BrainPolicy, eval_env: VecFlyAddictionEnv, mean_theta: np.ndarray,
                          n_substeps: int, eval_seed: int):
    """Evaluate the noiseless mean parameters on the taxis fitness (Task C)."""
    theta = np.tile(mean_theta.astype(np.float32), (eval_policy.batch, 1))
    return run_taxis_generation(eval_policy, eval_env, theta, seed=eval_seed, n_substeps=n_substeps)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["welfare", "hijacked", "taxis"], required=True)
    parser.add_argument("--gens", type=int, default=150)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.05,
                         help="ES mutation sigma for the decoder + 12 contact/intero encoder "
                              "params (spec: 0.05).")
    parser.add_argument("--sigma-steer", type=float, default=0.3,
                         help="LARGER ES mutation sigma for the 19 steering-encoder params "
                              "(a/c/m/h/b -- O(1) quantities, spec: 0.3). Per-parameter sigma "
                              "scaling (Task 3); see flyrl.policy.steering_param_sigma_vector.")
    parser.add_argument("--sigma-decay", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--lr-steer", type=float, default=0.2,
                         help="Task 4: LARGER Adam lr for the 19 steering-encoder params "
                              "(a/c/m/h/b -- O(1)-scale quantities that must travel ~3 units, "
                              "e.g. c_smoke 3->6) than --lr for the decoder + 12 contact/intero "
                              "encoder params. Adam moves each parameter by at most ~lr per "
                              "generation, so a single shared lr badly under-trains the "
                              "steering params relative to their larger --sigma-steer. Built "
                              "into a (P,) per-parameter lr vector the same way as "
                              "--sigma/--sigma-steer -- see steering_param_sigma_vector.")
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--steps-per-action", type=int, default=20)
    parser.add_argument("--beta-nic", type=float, default=1.0)
    parser.add_argument("--beta-jackpot", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=18)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=5,
                         help="spec (Task 3): evaluate the noiseless mean every 5 generations.")
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-from", type=str, default=None,
                         help="Path to another run's ckpt.npz (e.g. a --mode taxis checkpoint) "
                              "whose mean_theta seeds this run's starting parameters. Ignored if "
                              "--resume finds an existing checkpoint for THIS run.")
    parser.add_argument("--taxis-substeps", type=int, default=120,
                         help="--mode taxis only: steps per modality sub-episode (spec: 120).")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)

    run_dir = RESULTS_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"
    eval_path = run_dir / "eval.csv"
    config_path = run_dir / "config.json"

    assert args.population % 2 == 0, "--population must be even (antithetic pairs)"
    half = args.population // 2

    if args.resume and (run_dir / "ckpt.npz").exists():
        start_gen, mean_theta, adam, config = load_checkpoint(run_dir)
        config["gens"] = args.gens  # allow extending the run
        args = argparse.Namespace(**config)
        print(f"Resumed from {run_dir/'ckpt.npz'} at gen {start_gen}")
    else:
        config = vars(args).copy()
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        if args.init_from:
            init_theta = np.load(args.init_from)["mean_theta"].astype(np.float64)
            assert init_theta.shape == (N_PARAMS,), (
                f"--init-from {args.init_from} has mean_theta shape {init_theta.shape}, "
                f"expected ({N_PARAMS},)"
            )
            mean_theta = init_theta
            print(f"Initialized mean_theta from {args.init_from}")
        else:
            mean_theta = default_params(seed=args.seed).astype(np.float64)
        # Task 4: per-parameter Adam lr, built exactly like sigma_vec below
        # (reusing steering_param_sigma_vector) -- lr_steer for the 19
        # steering params, lr for everything else.
        lr_vec = steering_param_sigma_vector(args.lr, args.lr_steer, mean_theta.shape[0])
        adam = Adam(mean_theta.shape[0], lr=lr_vec)
        start_gen = 0

    # Task 3: per-parameter ES sigma -- the 19 steering-encoder params (a, c,
    # m, h, b) get a LARGER sigma (spec: 0.3) than the decoder + 12 contact/
    # intero encoder params (spec: 0.05), since the steering params are O(1)
    # quantities and where preference lives; --sigma alone would either
    # barely move them or blow up the decoder.
    sigma_vec = steering_param_sigma_vector(args.sigma, args.sigma_steer, mean_theta.shape[0])
    sigma_vec = sigma_vec * (args.sigma_decay ** start_gen)

    print(f"BrainPolicy(batch={args.population}) ... (loading brain weights)")
    policy = BrainPolicy(batch=args.population, device=args.device, dt=args.dt,
                          steps_per_action=args.steps_per_action, seed=args.seed)
    vec_env = VecFlyAddictionEnv(num_envs=args.population, n_steps=args.n_steps)
    assert policy.n_params == N_PARAMS
    print(f"n_params={policy.n_params}  n_input_neurons={policy.n_input_neurons}  "
          f"n_readout={policy.n_readout}  n_dan={policy.n_dan}")

    eval_policy = eval_env = None
    if args.eval_every > 0:
        eval_policy = BrainPolicy(batch=args.eval_episodes, device=args.device, dt=args.dt,
                                   steps_per_action=args.steps_per_action, seed=args.seed + 1)
        eval_env = VecFlyAddictionEnv(num_envs=args.eval_episodes, n_steps=args.n_steps)

    is_taxis = args.mode == "taxis"

    for gen in range(start_gen, args.gens):
        t0 = time.time()
        noise_rng = np.random.default_rng([args.seed, gen])
        theta_pop, eps_half = build_population(mean_theta, sigma_vec, half, noise_rng)

        env_seed = args.seed + 1_000_000 + gen
        if is_taxis:
            result = run_taxis_generation(policy, vec_env, theta_pop, seed=env_seed,
                                           n_substeps=args.taxis_substeps)
        else:
            result = run_episode_batch(policy, vec_env, theta_pop, seed=env_seed, n_steps=args.n_steps,
                                        mode=args.mode, beta_nic=args.beta_nic,
                                        beta_jackpot=args.beta_jackpot, gamma=args.gamma,
                                        common_seed=True)

        grad = es_gradient(eps_half, result.fitness, sigma_vec)
        update = adam.step(grad)
        mean_theta = mean_theta + update
        mean_theta = clip_steering_params(mean_theta)
        sigma_vec = sigma_vec * args.sigma_decay

        wall_s = time.time() - t0
        if is_taxis:
            tsum = _summarize_taxis(result)
            row = {
                "gen": gen, "wall_s": wall_s,
                "fitness_mean": float(result.fitness.mean()), "fitness_max": float(result.fitness.max()),
                **tsum,
            }
            _append_csv_row(log_path, TAXIS_LOG_COLUMNS, row)
            print(f"[{args.run_name}] gen {gen:4d}  wall={wall_s:6.2f}s  "
                  f"fitness={row['fitness_mean']:8.3f}/{row['fitness_max']:8.3f}  "
                  f"reach food={row['reach_food']:.2f} smoke={row['reach_smoke']:.2f} reels={row['reach_reels']:.2f}  "
                  f"dist_red food={row['dist_reduction_food']:.4f} smoke={row['dist_reduction_smoke']:.4f} "
                  f"reels={row['dist_reduction_reels']:.4f}")
        else:
            msum = _summarize_metrics(result.metrics)
            row = {
                "gen": gen, "wall_s": wall_s,
                "fitness_mean": float(result.fitness.mean()), "fitness_max": float(result.fitness.max()),
                "welfare_mean": float(result.welfare.mean()), "welfare_max": float(result.welfare.max()),
                "mean_dan_rate_hz": float(result.mean_dan_rate.mean()),
                "mean_dan_rate_driven_hz": float(result.mean_dan_rate_driven.mean()),
                **msum,
            }
            _append_csv_row(log_path, LOG_COLUMNS, row)
            print(f"[{args.run_name}] gen {gen:4d}  wall={wall_s:6.2f}s  "
                  f"fitness={row['fitness_mean']:8.3f}/{row['fitness_max']:8.3f}  "
                  f"welfare={row['welfare_mean']:8.3f}/{row['welfare_max']:8.3f}  "
                  f"food={row['frac_food']:.2f} smoke={row['frac_smoke']:.2f} reels={row['frac_reels']:.2f}  "
                  f"dan={row['mean_dan_rate_hz']:.1f}Hz/{row['mean_dan_rate_driven_hz']:.1f}Hz")

        if (gen + 1) % args.checkpoint_every == 0 or gen == args.gens - 1:
            save_checkpoint(run_dir, gen + 1, mean_theta, adam, config)

        if args.eval_every > 0 and (gen + 1) % args.eval_every == 0:
            if is_taxis:
                eval_result = _evaluate_mean_taxis(eval_policy, eval_env, mean_theta,
                                                    n_substeps=args.taxis_substeps, eval_seed=123456)
                etsum = _summarize_taxis(eval_result)
                erow = {
                    "gen": gen, "fitness_mean": float(eval_result.fitness.mean()),
                    "fitness_max": float(eval_result.fitness.max()), **etsum,
                }
                _append_csv_row(eval_path, TAXIS_EVAL_COLUMNS, erow)
                print(f"  [eval] gen {gen:4d}  fitness={erow['fitness_mean']:8.3f}/{erow['fitness_max']:8.3f}  "
                      f"reach food={erow['reach_food']:.2f} smoke={erow['reach_smoke']:.2f} "
                      f"reels={erow['reach_reels']:.2f}")
            else:
                eval_result = _evaluate_mean(eval_policy, eval_env, mean_theta, n_steps=args.n_steps,
                                              mode=args.mode, beta_nic=args.beta_nic,
                                              beta_jackpot=args.beta_jackpot, gamma=args.gamma,
                                              eval_seed=123456)
                emsum = _summarize_metrics(eval_result.metrics)
                steer_vals = unpack_steering_params(mean_theta)
                erow = {
                    "gen": gen,
                    "welfare_mean": float(eval_result.welfare.mean()),
                    "welfare_max": float(eval_result.welfare.max()),
                    "fitness_mean": float(eval_result.fitness.mean()),
                    "fitness_max": float(eval_result.fitness.max()),
                    "mean_dan_rate_hz": float(eval_result.mean_dan_rate.mean()),
                    "mean_dan_rate_driven_hz": float(eval_result.mean_dan_rate_driven.mean()),
                    **emsum, **steer_vals,
                }
                _append_csv_row(eval_path, EVAL_COLUMNS, erow)
                print(f"  [eval] gen {gen:4d}  welfare={erow['welfare_mean']:8.3f}  "
                      f"fitness={erow['fitness_mean']:8.3f}  "
                      f"food={erow['frac_food']:.2f} smoke={erow['frac_smoke']:.2f} reels={erow['frac_reels']:.2f}  "
                      f"a={[round(steer_vals[f'a_{s}'],2) for s in ('food','smoke','reels')]}  "
                      f"c={[round(steer_vals[f'c_{s}'],2) for s in ('food','smoke','reels')]}")

    save_checkpoint(run_dir, args.gens, mean_theta, adam, config)
    print(f"Training complete: {args.gens} generations. Final checkpoint: {run_dir/'ckpt.npz'}")


if __name__ == "__main__":
    main()
