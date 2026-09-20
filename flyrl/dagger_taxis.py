"""Task 2: linear-decodability test + DAgger taxis pretraining.

CLI
---
    python -m flyrl.dagger_taxis --run-name taxis_dagger

Rationale (see spec): ES found no learning signal for taxis in v2. Before
spending more compute on ES, this module asks a narrower question: is the
steering information linearly decodable from the (frozen-brain) descending-
neuron readout features AT ALL? The ENCODER is held FIXED at its
initialization throughout (v4: the 19-param shared steer_L/steer_R block +
the 6 own-channel contact/intero groups, i.e. exactly
``flyrl.policy.default_params()``'s encoder block) -- only the DECODER
(2 x flyrl.policy.N_FEATURES + 2 params -- 64, or 128 if
``flyrl.policy.USE_SLOW_TRACE``, +1 more if ``ADD_NO_SIGNAL_FEATURE``) is
fit, by DAgger + ridge regression onto the scripted teacher's action.

Algorithm
---------
Per modality (food/smoke/reels), a 120-step sub-episode with the env's obs
masked to that modality's own channels (flyrl.taxis.mask_obs), exactly as
--mode taxis. B=32 parallel envs, each with a DIFFERENT seed (unlike ES's
common-random-numbers population -- there is only one decoder here, not a
population, so CRN buys nothing).

~8 DAgger iterations, with a shrinking teacher-mixture probability beta:

    iter:  0     1     2     3     4     5     6     7
    beta:  1.0   0.5   0.3   0.2   0.1   0.0   0.0   0.0

Iteration 0 rolls out the TEACHER's actions only (obs still runs through the
fixed encoder + frozen brain to record readout features -- the features
returned by BrainPolicy.act()/features() depend only on obs, never on the
decoder, so this is a legitimate way to seed the dataset). Later iterations
roll out ``teacher if U(0,1) < beta else current-decoder-learner`` PER ENV
PER STEP, using the decoder fit after the previous iteration (a placeholder
zero decoder for the very first, beta=1 iteration, where it has no effect
on which action is taken).

Each iteration's 32 envs are split 24 train / 8 held-out-by-ENV (fixed
index split, never by timestep, to avoid within-episode leakage) BEFORE any
data is collected. (features_t, teacher_action_t) pairs from the 24 train
envs are appended to an ever-growing pool across iterations (DAgger
aggregation); the 8 held-out envs' pairs are used ONLY to choose ridge
lambda (grid search) and to report that iteration's held-out R^2 -- never to
fit weights. The final decoder (iteration 7) is assembled with the fixed
encoder into a full BrainPolicy-compatible theta vector and saved to
ckpt.npz's ``mean_theta``.

Two controls are also fit/run once at the end, using the same aggregated
data / eval seeds:
  - "bypass": ridge decoder fit on the raw encoder group-rates (12,) passed
    through the identical tau=50ms leaky-trace recursion the brain's
    readout uses, INSTEAD of the frozen brain's readout features -- the
    ceiling of a brain-less linear policy on this task.
  - "swap": the FINAL fitted decoder + brain, but with the target
    modality's L/R obs channels swapped before they reach the encoder at
    TEST time (physics/position tracking is unaffected) -- if the brain
    pathway really carries lateral information, the fly should steer AWAY
    from the target (reach rate/distance-reduction should get WORSE than
    even the forward-only baseline), not merely lose performance.
  - A latency ablation (tau=50ms vs a concatenated tau=50ms+tau=200ms
    feature set) is also tried on the same aggregated data; see the printed
    "slow_trace_ablation" block and controls.json's "adopted" flag for
    whether it clearly helped (>0.03 mean R^2). NOTE: even if it helps, the
    checkpoint's decoder stays 64-dim/tau=50ms-only for BrainPolicy
    compatibility -- see the run's final report for this explicit tradeoff.

This module is a thin, additive, importing-only consumer of
flyrl.addiction_env, flyrl.policy, flyrl.taxis and flyrl.scripted: none of
those files (nor flyrl.fastbrain) are modified by anything below except the
small, backward-compatible feature-exposing additions already made to
flyrl.policy.BrainPolicy (features()/features_slow_raw()/last_pooled*).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from flyrl.addiction_env import VecFlyAddictionEnv
from flyrl.policy import (
    BrainPolicy, default_params, _load_readout_cache,
    N_PARAMS, N_ENC_PARAMS, N_GROUPS, N_POOLS, N_FEATURES, N_ACTIONS,
)
from flyrl.taxis import MODALITIES, MODALITY_KEEP_CHANNELS, mask_obs, _distances
from flyrl.scripted import _steer_towards, _SOURCE_CHANNELS

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

TARGET_CLIP = 0.97

BETA_SCHEDULE = [1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0]

LOG_COLUMNS = [
    "iter", "beta", "n_train_samples", "lambda_turn_forward",
    "r2_turn", "r2_forward", "sign_agreement_turn",
    "reach_food", "reach_smoke", "reach_reels",
    "dist_reduction_food", "dist_reduction_smoke", "dist_reduction_reels",
    "wall_s",
]

EVAL_SEED_BASE = 900_000     # disjoint from all training-rollout seed ranges below
N_EVAL_ENVS = 32
LAMBDA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


# ---------------------------------------------------------------------
# Teacher / baseline actions
# ---------------------------------------------------------------------

def teacher_action_batch(obs: np.ndarray, infos: list, modality: str) -> np.ndarray:
    """(B,2) scripted-teacher action for every env, reusing
    flyrl.scripted._steer_towards's gain and lost-signal search exactly."""
    left_key, right_key = _SOURCE_CHANNELS[modality]
    B = obs.shape[0]
    out = np.zeros((B, 2), dtype=np.float32)
    for i in range(B):
        info_i = infos[i] if infos[i] else None
        out[i] = _steer_towards(obs[i], info_i, left_key, right_key, target_name=modality)
    return out


def forward_only_action_batch(B: int) -> np.ndarray:
    """Constant [turn=0, forward=+1] -- ignores all sensory input, walks
    straight ahead at max speed. The "forward-only baseline" referenced in
    the task's established facts / the Task 3 gate."""
    return np.tile(np.array([0.0, 1.0], dtype=np.float32), (B, 1))


# ---------------------------------------------------------------------
# Bypass-the-brain control: raw encoder group rates through the SAME
# tau=50ms leaky-trace recursion the brain's readout uses (closed form
# over `steps_per_action` identical sub-steps, since group_rates is
# constant within one macro-step).
# ---------------------------------------------------------------------

class BypassTracer:
    def __init__(self, batch: int, dt: float, steps_per_action: int, tau_ms: float = 50.0):
        decay_sub = float(np.exp(-dt / tau_ms))
        self.decay_macro = decay_sub ** steps_per_action
        self.gain = (dt / 1000.0) * (
            steps_per_action if decay_sub >= 1.0 - 1e-12
            else (1.0 - self.decay_macro) / (1.0 - decay_sub)
        )
        self.batch = batch
        self.trace = np.zeros((batch, N_GROUPS), dtype=np.float64)

    def reset(self):
        self.trace[:] = 0.0

    def step(self, group_rates: np.ndarray) -> np.ndarray:
        """group_rates: (B, N_GROUPS) Hz -> updated (B, N_GROUPS) trace."""
        self.trace = self.trace * self.decay_macro + group_rates * self.gain
        return self.trace.copy()


# ---------------------------------------------------------------------
# Ridge regression (closed form, with intercept, features NOT penalized
# for centering)
# ---------------------------------------------------------------------

def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float):
    Xm = X.mean(axis=0)
    Ym = Y.mean(axis=0)
    Xc = X - Xm
    Yc = Y - Ym
    F = X.shape[1]
    A = Xc.T @ Xc + lam * np.eye(F)
    Bmat = Xc.T @ Yc
    W = np.linalg.solve(A, Bmat)          # (F, K)
    b = Ym - Xm @ W                        # (K,)
    return W.astype(np.float64), b.astype(np.float64)


def ridge_predict(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return X @ W + b


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-8)


def choose_lambda_and_fit(X_train, Y_train, X_val, Y_val, grid=LAMBDA_GRID):
    best = None
    for lam in grid:
        W, b = ridge_fit(X_train, Y_train, lam)
        pred_val = ridge_predict(X_val, W, b)
        r2 = r2_score(Y_val, pred_val)
        score = float(np.mean(r2))
        if best is None or score > best[0]:
            best = (score, lam, W, b, r2)
    return best  # (mean_r2, lambda, W, b, r2_per_output)


def sign_agreement(pred_turn: np.ndarray, teacher_turn: np.ndarray, deadzone: float = 0.02) -> float:
    mask = np.abs(teacher_turn) > deadzone
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(pred_turn[mask]) == np.sign(teacher_turn[mask])))


# ---------------------------------------------------------------------
# Theta assembly (fixed encoder + a decoder we fit ourselves)
# ---------------------------------------------------------------------

def fixed_encoder_block(seed: int = 0) -> np.ndarray:
    """The encoder weight+bias block of flyrl.policy.default_params(),
    held FIXED throughout Task 2 (own-channel drive + bilateral contrast)."""
    return default_params(seed=seed)[:N_ENC_PARAMS].copy()


def assemble_theta(enc_block: np.ndarray, W_dec: np.ndarray, b_dec: np.ndarray) -> np.ndarray:
    """W_dec: (N_FEATURES, N_ACTIONS) as returned by ridge_fit; b_dec: (N_ACTIONS,).
    N_FEATURES == policy.N_FEATURES (64, or 128 if flyrl.policy.USE_SLOW_TRACE,
    +1 more if flyrl.policy.ADD_NO_SIGNAL_FEATURE) -- W_dec's row count MUST
    match whatever flyrl.policy is currently configured for. Returns a flat
    (N_PARAMS,) theta matching flyrl.policy's layout exactly."""
    assert W_dec.shape[0] == N_FEATURES, (
        f"W_dec has {W_dec.shape[0]} feature rows, but flyrl.policy.N_FEATURES={N_FEATURES} "
        "(USE_SLOW_TRACE/ADD_NO_SIGNAL_FEATURE mismatch between fit-time and current policy.py config)"
    )
    W_dec_policy = W_dec.T.astype(np.float32)  # (N_ACTIONS, N_FEATURES), policy.py's layout
    theta = np.concatenate([enc_block.astype(np.float32), W_dec_policy.reshape(-1), b_dec.astype(np.float32)])
    assert theta.shape == (N_PARAMS,)
    return theta


def zero_decoder_Wb():
    return np.zeros((N_FEATURES, N_ACTIONS), dtype=np.float64), np.zeros((N_ACTIONS,), dtype=np.float64)


# ---------------------------------------------------------------------
# One iteration's data-collection rollout
# ---------------------------------------------------------------------

class IterationData:
    __slots__ = ("X_fast", "X_slow_raw", "Y_raw", "reach", "dist_reduction")

    def __init__(self, X_fast, X_slow_raw, Y_raw, reach, dist_reduction):
        self.X_fast = X_fast          # (n, N_POOLS) z-normed tau=50ms features
        self.X_slow_raw = X_slow_raw  # (n, N_POOLS) raw tau=200ms features
        self.Y_raw = Y_raw            # (n, 2) raw teacher action (pre-clip/atanh)
        self.reach = reach            # dict[modality] -> (B,) bool (this rollout's own reach, diagnostic only)
        self.dist_reduction = dist_reduction


def collect_iteration(policy: BrainPolicy, vec_env: VecFlyAddictionEnv,
                       enc_block: np.ndarray, W_dec: np.ndarray, b_dec: np.ndarray,
                       beta: float, seed: int, n_substeps: int, rng: np.random.Generator) -> IterationData:
    """Roll out `beta`-mixture actions across all 3 modalities' sub-episodes
    (B envs each, DIFFERENT per-env seeds -- no CRN, there is no population
    here). Always records (features, teacher_action) regardless of which
    action was actually executed (that's the DAgger aggregation step)."""
    B = vec_env.num_envs
    theta = np.tile(assemble_theta(enc_block, W_dec, b_dec), (B, 1))
    policy.set_params(theta)

    X_fast_all, X_slow_all, Y_all = [], [], []
    reach = {}
    dist_reduction = {}

    for mi, modality in enumerate(MODALITIES):
        policy.reset()
        mod_seed = seed + mi * 7919
        obs = vec_env.reset(seed=mod_seed)  # per-env DIFFERENT seeds (seed + i)
        infos = [None] * B
        obs_masked = mask_obs(obs, modality)

        init_dist = _distances(vec_env, modality)
        r_consume = vec_env.envs[0].params.r_consume
        reached = np.zeros(B, dtype=bool)

        for _ in range(n_substeps):
            action_learner = policy.act(obs_masked)      # (B,2), also updates policy.last_pooled*
            features_fast = policy.features()             # (B, N_POOLS) z-normed tau=50ms
            features_slow = policy.features_slow_raw()    # (B, N_POOLS) raw tau=200ms

            teacher_action = teacher_action_batch(obs_masked, infos, modality)

            use_teacher = rng.uniform(size=B) < beta
            actual_action = np.where(use_teacher[:, None], teacher_action, action_learner).astype(np.float32)

            X_fast_all.append(features_fast.copy())
            X_slow_all.append(features_slow.copy())
            Y_all.append(teacher_action.copy())

            obs, _r, _d, infos = vec_env.step(actual_action)
            obs_masked = mask_obs(obs, modality)
            dist = _distances(vec_env, modality)
            reached |= dist <= r_consume

        final_dist = _distances(vec_env, modality)
        reach[modality] = reached
        dist_reduction[modality] = init_dist - final_dist

    X_fast = np.concatenate(X_fast_all, axis=0)
    X_slow = np.concatenate(X_slow_all, axis=0)
    Y = np.concatenate(Y_all, axis=0)
    return IterationData(X_fast, X_slow, Y, reach, dist_reduction)


# ---------------------------------------------------------------------
# Pure-policy evaluation rollout (teacher-only / forward-only / learner-only
# / L-R-swapped-learner), on FIXED eval seeds disjoint from training.
# ---------------------------------------------------------------------

def eval_rollout(mode: str, modality: str, n_substeps: int, seed_base: int = EVAL_SEED_BASE,
                  n_envs: int = N_EVAL_ENVS, policy: BrainPolicy = None,
                  enc_block: np.ndarray = None, W_dec: np.ndarray = None, b_dec: np.ndarray = None,
                  swap_lr: bool = False):
    """mode: 'teacher' | 'forward' | 'learner'. For 'learner', `policy`,
    `enc_block`, `W_dec`, `b_dec` must be given; if swap_lr, the modality's
    L/R obs channels are swapped before the encoder sees them (test-time
    only -- vec_env's own physics/position tracking is unaffected)."""
    vec_env = VecFlyAddictionEnv(num_envs=n_envs, n_steps=300)
    left_i, right_i = MODALITY_KEEP_CHANNELS[modality][0], MODALITY_KEEP_CHANNELS[modality][1]

    if mode == "learner":
        B = n_envs
        theta = np.tile(assemble_theta(enc_block, W_dec, b_dec), (B, 1))
        policy.set_params(theta)
        policy.reset()

    obs = vec_env.reset(seed=seed_base)
    infos = [None] * n_envs
    obs_masked = mask_obs(obs, modality)

    init_dist = _distances(vec_env, modality)
    r_consume = vec_env.envs[0].params.r_consume
    reached = np.zeros(n_envs, dtype=bool)

    for _ in range(n_substeps):
        if mode == "teacher":
            action = teacher_action_batch(obs_masked, infos, modality)
        elif mode == "forward":
            action = forward_only_action_batch(n_envs)
        elif mode == "learner":
            brain_in = obs_masked.copy()
            if swap_lr:
                brain_in[:, [left_i, right_i]] = brain_in[:, [right_i, left_i]]
            action = policy.act(brain_in)
        else:
            raise ValueError(mode)

        obs, _r, _d, infos = vec_env.step(action)
        obs_masked = mask_obs(obs, modality)
        dist = _distances(vec_env, modality)
        reached |= dist <= r_consume

    final_dist = _distances(vec_env, modality)
    return {
        "reach_rate": float(reached.mean()),
        "mean_dist_reduction": float(np.mean(init_dist - final_dist)),
    }


def diagnose_no_signal_search(policy: BrainPolicy, modality: str, enc_block: np.ndarray,
                               W_dec: np.ndarray, b_dec: np.ndarray, n_substeps: int = 300,
                               n_envs: int = N_EVAL_ENVS, seed_base: int = EVAL_SEED_BASE) -> dict:
    """Task 2: does the learner ROTATE TO SEARCH (like the scripted teacher's
    own `max(L,R) < lost_signal_eps -> spin in place` fallback,
    flyrl.scripted._steer_towards) when it has NO directional cue at all for
    `modality` -- e.g. reels outside its +-120deg field of view? Reports the
    fraction of steps with no signal and the mean |turn action| conditioned
    on no-signal vs signal (a searching policy should show |turn| well above
    0 -- ideally near the teacher's 1.0 -- specifically on no-signal steps)."""
    vec_env = VecFlyAddictionEnv(num_envs=n_envs, n_steps=300)
    B = n_envs
    theta = np.tile(assemble_theta(enc_block, W_dec, b_dec), (B, 1))
    policy.set_params(theta)
    policy.reset()

    obs = vec_env.reset(seed=seed_base)
    obs_masked = mask_obs(obs, modality)
    no_signal_turns, signal_turns, no_signal_frac = [], [], []

    for _ in range(n_substeps):
        action = policy.act(obs_masked)
        no_signal = policy.no_signal_feature()[:, 0] > 0.5  # (B,) bool
        no_signal_frac.append(float(no_signal.mean()))
        if no_signal.any():
            no_signal_turns.append(np.abs(action[no_signal, 0]))
        if (~no_signal).any():
            signal_turns.append(np.abs(action[~no_signal, 0]))
        obs, _r, _d, _infos = vec_env.step(action)
        obs_masked = mask_obs(obs, modality)

    ns_turns = np.concatenate(no_signal_turns) if no_signal_turns else np.zeros(0)
    s_turns = np.concatenate(signal_turns) if signal_turns else np.zeros(0)
    return {
        "frac_steps_no_signal": float(np.mean(no_signal_frac)),
        "mean_abs_turn_when_no_signal": float(ns_turns.mean()) if ns_turns.size else float("nan"),
        "mean_abs_turn_when_signal": float(s_turns.mean()) if s_turns.size else float("nan"),
        "n_no_signal_samples": int(ns_turns.size),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, default="taxis_dagger")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--val-envs", type=int, default=8, help="of --population, held out by ENV each iteration")
    parser.add_argument("--n-substeps", type=int, default=120)
    parser.add_argument("--eval-envs", type=int, default=N_EVAL_ENVS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--steps-per-action", type=int, default=20)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)

    run_dir = RESULTS_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"
    controls_path = run_dir / "controls.json"
    ckpt_path = run_dir / "ckpt.npz"
    if log_path.exists():
        log_path.unlink()

    B = args.population
    n_val = args.val_envs
    n_train_envs = B - n_val
    assert 0 < n_val < B

    enc_block = fixed_encoder_block(seed=args.seed)
    policy = BrainPolicy(batch=B, device=args.device, dt=args.dt,
                          steps_per_action=args.steps_per_action, seed=args.seed)
    eval_policy = BrainPolicy(batch=args.eval_envs, device=args.device, dt=args.dt,
                               steps_per_action=args.steps_per_action, seed=args.seed + 1)
    vec_env = VecFlyAddictionEnv(num_envs=B, n_steps=300)
    bypass = BypassTracer(batch=B, dt=args.dt, steps_per_action=args.steps_per_action)
    rng = np.random.default_rng(args.seed)

    # ---- one-off baselines (don't depend on iteration) ----
    print("Computing teacher-only / forward-only baselines on the fixed eval seeds ...")
    baseline = {"teacher": {}, "forward": {}}
    for modality in MODALITIES:
        baseline["teacher"][modality] = eval_rollout("teacher", modality, args.n_substeps,
                                                       n_envs=args.eval_envs)
        baseline["forward"][modality] = eval_rollout("forward", modality, args.n_substeps,
                                                       n_envs=args.eval_envs)
        print(f"  {modality:>6s}  teacher: reach={baseline['teacher'][modality]['reach_rate']:.3f} "
              f"dist_red={baseline['teacher'][modality]['mean_dist_reduction']:+.4f}   "
              f"forward: reach={baseline['forward'][modality]['reach_rate']:.3f} "
              f"dist_red={baseline['forward'][modality]['mean_dist_reduction']:+.4f}")

    W_dec, b_dec = zero_decoder_Wb()  # iteration-0 placeholder (irrelevant: beta=1.0)
    train_X_fast, train_X_slow, train_Y = [], [], []
    beta_schedule = (BETA_SCHEDULE + [0.0] * args.iterations)[:args.iterations]

    rows = []
    for it in range(args.iterations):
        t0 = time.time()
        beta = beta_schedule[it]
        iter_seed = 10_000 + it * 100_003 + args.seed

        data = collect_iteration(policy, vec_env, enc_block, W_dec, b_dec,
                                  beta=beta, seed=iter_seed, n_substeps=args.n_substeps, rng=rng)

        # env-level split: rows are laid out modality-major, then step, then
        # env (see collect_iteration's per-step X_fast_all.append order), so
        # row i's env index is (i % B).
        env_id = np.concatenate([np.tile(np.arange(B), args.n_substeps) for _ in MODALITIES])
        train_mask = env_id < n_train_envs
        val_mask = ~train_mask

        train_X_fast.append(data.X_fast[train_mask])
        train_X_slow.append(data.X_slow_raw[train_mask])
        train_Y.append(np.clip(data.Y_raw[train_mask], -TARGET_CLIP, TARGET_CLIP))

        X_train_pool = np.concatenate(train_X_fast, axis=0)
        Y_train_pool = np.arctanh(np.concatenate(train_Y, axis=0))

        X_val = data.X_fast[val_mask]
        Y_val_raw = np.clip(data.Y_raw[val_mask], -TARGET_CLIP, TARGET_CLIP)
        Y_val = np.arctanh(Y_val_raw)

        mean_r2, lam, W_dec, b_dec, r2_per_out = choose_lambda_and_fit(
            X_train_pool, Y_train_pool, X_val, Y_val)
        pred_val = ridge_predict(X_val, W_dec, b_dec)
        sign_agr = sign_agreement(pred_val[:, 0], Y_val_raw[:, 0])

        # learner-only rollout with the JUST-fit decoder
        reach = {}
        dist_red = {}
        for modality in MODALITIES:
            r = eval_rollout("learner", modality, args.n_substeps, n_envs=args.eval_envs,
                              policy=eval_policy, enc_block=enc_block, W_dec=W_dec, b_dec=b_dec)
            reach[modality] = r["reach_rate"]
            dist_red[modality] = r["mean_dist_reduction"]

        wall_s = time.time() - t0
        row = {
            "iter": it, "beta": beta, "n_train_samples": int(X_train_pool.shape[0]),
            "lambda_turn_forward": lam,
            "r2_turn": float(r2_per_out[0]), "r2_forward": float(r2_per_out[1]),
            "sign_agreement_turn": sign_agr,
            "reach_food": reach["food"], "reach_smoke": reach["smoke"], "reach_reels": reach["reels"],
            "dist_reduction_food": dist_red["food"], "dist_reduction_smoke": dist_red["smoke"],
            "dist_reduction_reels": dist_red["reels"],
            "wall_s": wall_s,
        }
        rows.append(row)
        new_file = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        print(f"[dagger] iter {it}  beta={beta:.2f}  n_train={row['n_train_samples']:6d}  lam={lam:8.3f}  "
              f"R2 turn={row['r2_turn']:+.4f} fwd={row['r2_forward']:+.4f}  "
              f"sign_agree={sign_agr:.3f}  "
              f"reach food={reach['food']:.2f} smoke={reach['smoke']:.2f} reels={reach['reels']:.2f}  "
              f"({wall_s:.1f}s)")

    # ---- final checkpoint ----
    final_theta = assemble_theta(enc_block, W_dec, b_dec)
    np.savez(ckpt_path, mean_theta=final_theta.astype(np.float32),
             gen=np.array(args.iterations),
             config_json=np.array(json.dumps(vars(args))))
    print(f"Saved final decoder theta to {ckpt_path}")

    # ---- controls ----
    print("Running controls (bypass-the-brain ceiling, L/R-swap sanity check, slow-trace ablation) ...")
    X_train_pool = np.concatenate(train_X_fast, axis=0)
    X_train_slow_pool = np.concatenate(train_X_slow, axis=0)
    Y_train_pool_raw = np.concatenate(train_Y, axis=0)
    Y_train_pool = np.arctanh(Y_train_pool_raw)

    # bypass control needs its OWN feature stream (group-rate traces, not
    # brain readout traces) -- collected via a dedicated rollout at beta=0.5
    # with the FINAL decoder (see collect_bypass_dataset), so its state
    # distribution is comparable to what the final brain-based decoder sees.
    bypass_X_all, bypass_Y_all = collect_bypass_dataset(
        policy, vec_env, bypass, enc_block, W_dec, b_dec, args.n_substeps, rng)
    n_train_b = int(0.75 * bypass_X_all.shape[0])
    idx_perm = rng.permutation(bypass_X_all.shape[0])
    tr_idx, va_idx = idx_perm[:n_train_b], idx_perm[n_train_b:]
    bx_mean, bx_std = bypass_X_all[tr_idx].mean(0), bypass_X_all[tr_idx].std(0) + 1e-6
    bXn = (bypass_X_all - bx_mean) / bx_std
    bY = np.arctanh(np.clip(bypass_Y_all, -TARGET_CLIP, TARGET_CLIP))
    b_mean_r2, b_lam, bW, bb, b_r2 = choose_lambda_and_fit(bXn[tr_idx], bY[tr_idx], bXn[va_idx], bY[va_idx])

    # slow-trace ablation: fast (z-normed via screen stats, already X_train_pool)
    # concatenated with slow (z-normed via the SAME kind of FIXED screen stats,
    # flyrl/readout_neurons.npz's pool_mean_slow/pool_std_slow -- consistent
    # with how it would actually be integrated into BrainPolicy if adopted,
    # unlike v3's train-pool-derived standardization).
    _, _, _, _, _, _, _, pool_mean_slow, pool_std_slow = _load_readout_cache()
    assert pool_mean_slow is not None, "flyrl/readout_neurons.npz missing pool_mean_slow -- rerun scripts/diag_readout.py"
    X_slow_norm = (X_train_slow_pool - pool_mean_slow) / pool_std_slow
    X_dual_train = np.concatenate([X_train_pool, X_slow_norm], axis=1)
    val_slow_norm = (data.X_slow_raw[val_mask] - pool_mean_slow) / pool_std_slow
    X_dual_val = np.concatenate([X_val, val_slow_norm], axis=1)
    dual_mean_r2, dual_lam, dW, db, dual_r2 = choose_lambda_and_fit(
        X_dual_train, Y_train_pool, X_dual_val, Y_val)
    fast_only_val_r2 = r2_score(Y_val, ridge_predict(X_val, W_dec, b_dec))
    slow_helps = bool(np.mean(dual_r2) > np.mean(fast_only_val_r2) + 0.03)

    # L/R swap sanity check with the FINAL decoder
    swap_results = {}
    normal_results = {}
    for modality in MODALITIES:
        normal_results[modality] = eval_rollout("learner", modality, args.n_substeps, n_envs=args.eval_envs,
                                                  policy=eval_policy, enc_block=enc_block, W_dec=W_dec, b_dec=b_dec,
                                                  swap_lr=False)
        swap_results[modality] = eval_rollout("learner", modality, args.n_substeps, n_envs=args.eval_envs,
                                               policy=eval_policy, enc_block=enc_block, W_dec=W_dec, b_dec=b_dec,
                                               swap_lr=True)

    # Task 2: evaluate the FINAL learner at the real episode length
    # (300 steps, not the 120-step sub-episodes used for DAgger training/
    # iteration reporting above), per modality.
    print("Evaluating final learner at 300-step sub-episodes (real episode length) ...")
    eval_300 = {}
    for modality in MODALITIES:
        eval_300[modality] = eval_rollout("learner", modality, 300, n_envs=args.eval_envs,
                                           policy=eval_policy, enc_block=enc_block, W_dec=W_dec, b_dec=b_dec,
                                           swap_lr=False)
        print(f"  {modality:>6s}  300-step reach={eval_300[modality]['reach_rate']:.3f}  "
              f"dist_red={eval_300[modality]['mean_dist_reduction']:+.4f}")

    # Task 2: does the learner rotate to search when it has no directional
    # signal at all (reels' limited +-120deg FOV is the main suspect)?
    print("Diagnosing no-signal search behaviour (300-step) per modality ...")
    no_signal_diag = {}
    for modality in MODALITIES:
        no_signal_diag[modality] = diagnose_no_signal_search(
            eval_policy, modality, enc_block, W_dec, b_dec, n_substeps=300, n_envs=args.eval_envs)
        d = no_signal_diag[modality]
        print(f"  {modality:>6s}  frac_no_signal={d['frac_steps_no_signal']:.3f}  "
              f"|turn|_no_signal={d['mean_abs_turn_when_no_signal']:.3f}  "
              f"|turn|_signal={d['mean_abs_turn_when_signal']:.3f}")

    controls = {
        "bypass_brain_ceiling": {
            "lambda": b_lam, "r2_turn": float(b_r2[0]), "r2_forward": float(b_r2[1]),
            "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
            "note": f"ridge decoder fit on raw encoder group-rates ({N_GROUPS}-dim) passed through "
                    "the same tau=50ms leaky-trace recursion the brain readout uses, "
                    "BYPASSING the frozen brain entirely.",
        },
        "final_decoder_fast_trace_only_val_r2": {
            "r2_turn": float(fast_only_val_r2[0]), "r2_forward": float(fast_only_val_r2[1]),
        },
        "slow_trace_ablation": {
            "lambda": dual_lam, "r2_turn": float(dual_r2[0]), "r2_forward": float(dual_r2[1]),
            "adopted": slow_helps,
            "note": "tau=50ms (64, z-normed via fixed screen stats) concatenated with tau=200ms "
                    "(64, z-normed via fixed screen stats pool_mean_slow/pool_std_slow) -> 128 "
                    "features. If mean R^2 improves by > 0.03 over fast-trace-only, "
                    "flyrl.policy.USE_SLOW_TRACE must be set True and this run re-launched from "
                    "scratch so the DAgger decoder is actually fit at feature dim 128 (no "
                    "half-measures -- this script only REPORTS the ablation, it does not flip "
                    "the flag or refit itself).",
        },
        "lr_swap_sanity_check": {
            modality: {"normal": normal_results[modality], "swapped": swap_results[modality]}
            for modality in MODALITIES
        },
        "eval_300step": eval_300,
        "no_signal_search_diagnostic": no_signal_diag,
        "baselines": baseline,
    }
    with open(controls_path, "w") as f:
        json.dump(controls, f, indent=2)
    print(f"Saved {controls_path}")
    print(json.dumps(controls, indent=2))

    gate_reach = np.mean([rows[-1]["reach_food"], rows[-1]["reach_smoke"], rows[-1]["reach_reels"]])
    fwd_reach = np.mean([baseline["forward"][m]["reach_rate"] for m in MODALITIES])
    print(f"\nFinal learner-only mean reach rate: {gate_reach:.3f}  "
          f"forward-only baseline mean reach rate: {fwd_reach:.3f}  "
          f"ratio: {gate_reach / max(fwd_reach, 1e-9):.2f}x")
    return rows, controls


def collect_bypass_dataset(policy, vec_env, bypass, enc_block, W_dec, b_dec, n_substeps, rng,
                            seed=777_001, beta=0.5):
    """A standalone rollout (beta=0.5 mixture with the FINAL decoder) purely
    to build the bypass-control's (group-rate-trace, teacher-action)
    dataset -- kept separate from the main aggregation loop for clarity."""
    B = vec_env.num_envs
    theta = np.tile(assemble_theta(enc_block, W_dec, b_dec), (B, 1))
    policy.set_params(theta)
    X_all, Y_all = [], []
    for mi, modality in enumerate(MODALITIES):
        policy.reset()
        bypass.reset()
        obs = vec_env.reset(seed=seed + mi * 7919)
        infos = [None] * B
        obs_masked = mask_obs(obs, modality)
        for _ in range(n_substeps):
            action_learner = policy.act(obs_masked)
            group_rates = policy.group_rates(obs_masked)
            bypass_feat = bypass.step(group_rates)
            teacher_action = teacher_action_batch(obs_masked, infos, modality)
            use_teacher = rng.uniform(size=B) < beta
            actual_action = np.where(use_teacher[:, None], teacher_action, action_learner).astype(np.float32)
            X_all.append(bypass_feat.copy())
            Y_all.append(teacher_action.copy())
            obs, _r, _d, infos = vec_env.step(actual_action)
            obs_masked = mask_obs(obs, modality)
    return np.concatenate(X_all, axis=0), np.concatenate(Y_all, axis=0)


if __name__ == "__main__":
    main()
