"""Fast tests for flyrl.dagger_taxis (Task 2). Tiny B / few steps throughout
-- designed to run in well under a minute total.

Covers: the teacher/forward-only action helpers, the closed-form bypass
trace (checked against an explicit step-by-step simulation), ridge
regression + R^2/sign-agreement plumbing on synthetic data, theta assembly
(BrainPolicy.set_params compatibility), and one tiny end-to-end
collect_iteration / eval_rollout smoke test through the real frozen brain.
"""

from __future__ import annotations

import numpy as np
import pytest

from flyrl.addiction_env import FlyAddictionEnv, VecFlyAddictionEnv
from flyrl.policy import BrainPolicy, N_PARAMS, N_POOLS, N_FEATURES, N_GROUPS, N_ACTIONS
from flyrl.taxis import MODALITIES, MODALITY_KEEP_CHANNELS
from flyrl.scripted import CHANNEL_INDEX, _SOURCE_CHANNELS
from flyrl.dagger_taxis import (
    teacher_action_batch, forward_only_action_batch, BypassTracer,
    ridge_fit, ridge_predict, r2_score, choose_lambda_and_fit, sign_agreement,
    fixed_encoder_block, assemble_theta, zero_decoder_Wb,
    collect_iteration, eval_rollout, N_ENC_PARAMS,
)


# ---------------------------------------------------------------------
# Teacher / baseline action helpers
# ---------------------------------------------------------------------

def test_forward_only_action_batch():
    B = 5
    action = forward_only_action_batch(B)
    assert action.shape == (B, 2)
    np.testing.assert_array_equal(action, np.tile([0.0, 1.0], (B, 1)))


def test_teacher_action_batch_steers_toward_stronger_side():
    B = 2
    obs = np.zeros((B, 12), dtype=np.float32)
    obs[0, CHANNEL_INDEX["food_odor_L"]] = 0.8
    obs[0, CHANNEL_INDEX["food_odor_R"]] = 0.1
    obs[1, CHANNEL_INDEX["food_odor_L"]] = 0.1
    obs[1, CHANNEL_INDEX["food_odor_R"]] = 0.8
    infos = [None, None]
    action = teacher_action_batch(obs, infos, "food")
    assert action.shape == (B, 2)
    assert action[0, 0] > 0.0, "stronger LEFT food signal should turn left (positive turn)"
    assert action[1, 0] < 0.0, "stronger RIGHT food signal should turn right (negative turn)"


def test_modality_keep_channels_LR_order_matches_scripted_source_channels():
    """Sanity check the L/R-swap control's assumption: MODALITY_KEEP_CHANNELS
    lists (left_idx, right_idx, ...) in the SAME order flyrl.scripted uses."""
    for modality in MODALITIES:
        left_key, right_key = _SOURCE_CHANNELS[modality]
        expected = [CHANNEL_INDEX[left_key], CHANNEL_INDEX[right_key]]
        assert MODALITY_KEEP_CHANNELS[modality][:2] == expected


# ---------------------------------------------------------------------
# Bypass trace: closed form must match an explicit step-by-step simulation
# ---------------------------------------------------------------------

def test_bypass_tracer_matches_explicit_simulation():
    dt, steps_per_action, tau_ms = 0.5, 20, 50.0
    B = 3
    rng = np.random.default_rng(0)
    rates = rng.uniform(0, 150, size=(B, N_GROUPS)).astype(np.float64)

    tracer = BypassTracer(batch=B, dt=dt, steps_per_action=steps_per_action, tau_ms=tau_ms)
    tracer.reset()
    got = tracer.step(rates)

    decay_sub = np.exp(-dt / tau_ms)
    manual = np.zeros((B, N_GROUPS), dtype=np.float64)
    for _ in range(steps_per_action):
        manual = manual * decay_sub + rates * (dt / 1000.0)

    np.testing.assert_allclose(got, manual, rtol=1e-8)

    # a second call should compound on the retained trace state
    got2 = tracer.step(rates)
    for _ in range(steps_per_action):
        manual = manual * decay_sub + rates * (dt / 1000.0)
    np.testing.assert_allclose(got2, manual, rtol=1e-8)


def test_bypass_tracer_reset_zeroes_state():
    tracer = BypassTracer(batch=2, dt=0.5, steps_per_action=20)
    tracer.step(np.full((2, N_GROUPS), 100.0))
    assert np.any(tracer.trace != 0.0)
    tracer.reset()
    np.testing.assert_array_equal(tracer.trace, np.zeros((2, N_GROUPS)))


# ---------------------------------------------------------------------
# Ridge regression / R^2 / sign agreement
# ---------------------------------------------------------------------

def test_ridge_fit_recovers_linear_relationship():
    rng = np.random.default_rng(0)
    n, F, K = 400, 10, 2
    X = rng.standard_normal((n, F))
    W_true = rng.standard_normal((F, K)) * 0.5
    b_true = rng.standard_normal(K)
    Y = X @ W_true + b_true + 0.01 * rng.standard_normal((n, K))

    W, b = ridge_fit(X, Y, lam=1e-3)
    np.testing.assert_allclose(W, W_true, atol=0.15)
    np.testing.assert_allclose(b, b_true, atol=0.1)

    pred = ridge_predict(X, W, b)
    r2 = r2_score(Y, pred)
    assert np.all(r2 > 0.95)


def test_r2_score_perfect_and_mean_baseline():
    y_true = np.array([[1.0, -1.0], [2.0, 0.0], [3.0, 1.0]])
    r2_perfect = r2_score(y_true, y_true.copy())
    np.testing.assert_allclose(r2_perfect, [1.0, 1.0], atol=1e-10)

    y_pred_mean = np.tile(y_true.mean(axis=0), (3, 1))
    r2_mean = r2_score(y_true, y_pred_mean)
    np.testing.assert_allclose(r2_mean, [0.0, 0.0], atol=1e-10)


def test_choose_lambda_and_fit_prefers_signal_over_noise():
    rng = np.random.default_rng(1)
    n, F = 300, 6
    X_train = rng.standard_normal((n, F))
    W_true = rng.standard_normal((F, 2))
    Y_train = X_train @ W_true + 0.05 * rng.standard_normal((n, 2))
    X_val = rng.standard_normal((100, F))
    Y_val = X_val @ W_true + 0.05 * rng.standard_normal((100, 2))

    mean_r2, lam, W, b, r2_per_out = choose_lambda_and_fit(X_train, Y_train, X_val, Y_val)
    assert mean_r2 > 0.9
    assert np.all(r2_per_out > 0.85)


def test_sign_agreement_basic():
    teacher = np.array([0.5, -0.3, 0.8, -0.9, 0.001])
    pred_same = np.array([0.2, -0.1, 0.9, -0.5, 5.0])  # last one within deadzone of teacher, ignored
    agr = sign_agreement(pred_same, teacher, deadzone=0.02)
    assert agr == pytest.approx(1.0)

    pred_flip = -teacher
    agr_flip = sign_agreement(pred_flip, teacher, deadzone=0.02)
    assert agr_flip == pytest.approx(0.0)


# ---------------------------------------------------------------------
# Theta assembly / BrainPolicy compatibility
# ---------------------------------------------------------------------

def test_assemble_theta_shape_and_settable():
    enc_block = fixed_encoder_block(seed=0)
    assert enc_block.shape == (N_ENC_PARAMS,)
    W_dec, b_dec = zero_decoder_Wb()
    assert W_dec.shape == (N_FEATURES, N_ACTIONS)
    theta = assemble_theta(enc_block, W_dec, b_dec)
    assert theta.shape == (N_PARAMS,)
    assert theta.dtype == np.float32

    B = 2
    pol = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=0)
    pol.set_params(np.tile(theta, (B, 1)))
    pol.reset()
    obs = np.zeros((B, 12), dtype=np.float32)
    action = pol.act(obs)
    assert action.shape == (B, 2)
    # zero decoder -> pre_dec == b_dec == [0, 0] -> tanh(0) == 0 exactly
    np.testing.assert_allclose(action, np.zeros((B, 2)), atol=1e-6)


def test_brainpolicy_features_shape_and_finite():
    B = 2
    pol = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=0)
    pol.reset()
    obs = np.random.default_rng(0).uniform(0, 1, size=(B, 12)).astype(np.float32)
    pol.act(obs)
    feats = pol.features()
    feats_slow = pol.features_slow_raw()
    assert feats.shape == (B, N_FEATURES)
    assert feats_slow.shape == (B, N_POOLS)
    assert np.all(np.isfinite(feats))
    assert np.all(np.isfinite(feats_slow))
    assert np.all(feats_slow >= 0.0), "raw trace of nonnegative spike counts must stay nonnegative"


# ---------------------------------------------------------------------
# Tiny end-to-end smoke tests through the real frozen brain
# ---------------------------------------------------------------------

def test_collect_iteration_tiny_shapes():
    B = 2
    n_substeps = 3
    policy = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=0)
    vec_env = VecFlyAddictionEnv(num_envs=B, n_steps=300)
    enc_block = fixed_encoder_block(seed=0)
    W_dec, b_dec = zero_decoder_Wb()
    rng = np.random.default_rng(0)

    data = collect_iteration(policy, vec_env, enc_block, W_dec, b_dec,
                              beta=1.0, seed=123, n_substeps=n_substeps, rng=rng)

    n_expected = n_substeps * B * len(MODALITIES)
    assert data.X_fast.shape == (n_expected, N_FEATURES)
    assert data.X_slow_raw.shape == (n_expected, N_POOLS)
    assert data.Y_raw.shape == (n_expected, 2)
    assert np.all(np.isfinite(data.X_fast))
    assert np.all(np.isfinite(data.X_slow_raw))
    assert np.all(np.isfinite(data.Y_raw))
    assert np.all(data.Y_raw >= -1.0) and np.all(data.Y_raw <= 1.0)
    for modality in MODALITIES:
        assert data.reach[modality].shape == (B,)
        assert data.dist_reduction[modality].shape == (B,)


def test_eval_rollout_forward_and_teacher_tiny():
    for mode in ("forward", "teacher"):
        result = eval_rollout(mode, "food", n_substeps=3, n_envs=4)
        assert 0.0 <= result["reach_rate"] <= 1.0
        assert np.isfinite(result["mean_dist_reduction"])


def test_eval_rollout_learner_tiny():
    B = 3
    policy = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=0)
    enc_block = fixed_encoder_block(seed=0)
    W_dec, b_dec = zero_decoder_Wb()
    result = eval_rollout("learner", "smoke", n_substeps=3, n_envs=B, policy=policy,
                           enc_block=enc_block, W_dec=W_dec, b_dec=b_dec, swap_lr=False)
    assert 0.0 <= result["reach_rate"] <= 1.0
    assert np.isfinite(result["mean_dist_reduction"])

    result_swapped = eval_rollout("learner", "smoke", n_substeps=3, n_envs=B, policy=policy,
                                   enc_block=enc_block, W_dec=W_dec, b_dec=b_dec, swap_lr=True)
    assert 0.0 <= result_swapped["reach_rate"] <= 1.0
    assert np.isfinite(result_swapped["mean_dist_reduction"])
