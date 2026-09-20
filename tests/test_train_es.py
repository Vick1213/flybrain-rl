"""Fast unit tests for the Task 4 per-parameter Adam lr (--lr-steer) and
steering-param clipping in flyrl.train_es.

Problem being tested: per-parameter ES sigma alone (sigma_steer=0.3 for the
19 O(1)-scale steering params vs sigma=0.05 for the rest) is not enough --
Adam moves each parameter by at most ~lr per generation REGARDLESS of
sigma, so a single shared --lr badly under-trains the steering params.
--lr-steer gives them a larger Adam lr, built the same way as sigma_vec
(steering_param_sigma_vector); clip_steering_params then keeps the
resulting bigger steps from running the encoder's sigmoid drives away.

No BrainPolicy/env rollout here -- everything is exercised through a
synthetic (hand-picked) fitness function and gradients, so this runs in a
fraction of a second.
"""

from __future__ import annotations

import numpy as np
import pytest

import json

from flyrl.policy import N_PARAMS, N_ENC_PARAMS, N_STEER_PARAMS, steering_param_names, steering_param_sigma_vector, default_params
from flyrl.train_es import (
    Adam, build_population, es_gradient, clip_steering_params,
    STEER_CLIP_B, STEER_CLIP_ACMH, EpisodeResult, METRIC_KEYS, _average_episode_results,
)
import flyrl.train_es as train_es


# ---------------------------------------------------------------------
# Adam accepts a (P,) per-parameter lr array
# ---------------------------------------------------------------------

def test_adam_lr_vector_shape_and_scalar_still_work():
    lr_vec = steering_param_sigma_vector(0.03, 0.2, N_PARAMS)
    adam = Adam(N_PARAMS, lr=lr_vec)
    assert isinstance(adam.lr, np.ndarray) and adam.lr.shape == (N_PARAMS,)

    adam_scalar = Adam(N_PARAMS, lr=0.03)
    assert isinstance(adam_scalar.lr, float)

    with pytest.raises(AssertionError):
        Adam(N_PARAMS, lr=np.ones(N_PARAMS - 1))  # wrong shape must be rejected


# ---------------------------------------------------------------------
# One Adam update moves a steering param by ~lr_steer and a decoder param
# by ~lr, via a real build_population -> es_gradient -> Adam.step pipeline
# driven by a synthetic fitness function (not the real env/brain).
# ---------------------------------------------------------------------

def test_one_update_moves_steering_param_by_lr_steer_and_decoder_by_lr():
    lr = 0.03
    lr_steer = 0.2
    assert lr_steer > 3 * lr, "test assumes lr_steer is clearly larger than lr"

    rng = np.random.default_rng(0)
    mean_theta = np.zeros(N_PARAMS, dtype=np.float64)
    sigma_vec = steering_param_sigma_vector(0.05, 0.3, N_PARAMS)
    half = 64

    steer_idx = steering_param_names().index("a_smoke")
    dec_idx = N_PARAMS - 1  # last param is in the decoder block

    theta_pop, eps_half = build_population(mean_theta, sigma_vec, half, rng)
    # Synthetic fitness: increasing in BOTH theta[steer_idx] and
    # theta[dec_idx] only (every other param has zero effect on fitness,
    # exactly like a real fitness landscape ES only sees through samples).
    fitness = theta_pop[:, steer_idx] + theta_pop[:, dec_idx]

    grad = es_gradient(eps_half, fitness, sigma_vec)
    assert grad[steer_idx] > 0 and grad[dec_idx] > 0, "ES gradient should point the right way"

    lr_vec = steering_param_sigma_vector(lr, lr_steer, N_PARAMS)
    adam = Adam(N_PARAMS, lr=lr_vec)
    update = adam.step(grad)

    # First Adam step is ~lr * sign(grad) (m_hat == grad, v_hat == grad**2
    # exactly at t=1, before bias-correction averaging kicks in), so the
    # step size is set almost entirely by lr/lr_steer, not by the gradient
    # magnitude -- this is exactly the property --lr-steer relies on.
    assert update[steer_idx] == pytest.approx(lr_steer, rel=0.05)
    assert update[dec_idx] == pytest.approx(lr, rel=0.05)
    assert np.all(np.isfinite(update))


# ---------------------------------------------------------------------
# Steering-param clipping
# ---------------------------------------------------------------------

def test_clip_steering_params_clamps_only_steering_block():
    theta = np.zeros(N_PARAMS, dtype=np.float64)
    names = steering_param_names()
    b_idx = names.index("b_steer")
    a_idx = names.index("a_food")
    c_idx = names.index("c_smoke")
    dec_idx = N_STEER_PARAMS + 1  # a non-steering (contact/decoder) param

    theta[b_idx] = 100.0     # should clamp to STEER_CLIP_B[1] = 2
    theta[a_idx] = -100.0    # should clamp to STEER_CLIP_ACMH[0] = -8
    theta[c_idx] = 5.0       # within [-8, 8]: unchanged
    theta[dec_idx] = 999.0   # non-steering: must NEVER be touched

    clipped = clip_steering_params(theta)

    assert clipped[b_idx] == pytest.approx(STEER_CLIP_B[1])
    assert clipped[a_idx] == pytest.approx(STEER_CLIP_ACMH[0])
    assert clipped[c_idx] == pytest.approx(5.0)
    assert clipped[dec_idx] == pytest.approx(999.0)
    # original array must not be mutated in place
    assert theta[b_idx] == pytest.approx(100.0)


def test_clip_steering_params_lower_bound_b_steer():
    theta = np.zeros(N_PARAMS, dtype=np.float64)
    b_idx = steering_param_names().index("b_steer")
    theta[b_idx] = -100.0
    clipped = clip_steering_params(theta)
    assert clipped[b_idx] == pytest.approx(STEER_CLIP_B[0])


# ---------------------------------------------------------------------
# Task D: es_gradient must not produce NaN for exact-zero-sigma entries
# (--freeze-decoder gives the decoder block sigma=0 AND lr=0)
# ---------------------------------------------------------------------

def test_es_gradient_zero_sigma_entries_give_zero_grad_not_nan():
    rng = np.random.default_rng(0)
    P = 6
    half = 8
    sigma = np.array([0.3, 0.3, 0.0, 0.0, 0.05, 0.05])  # last-4 mimic a frozen decoder tail... mixed here
    eps_half = rng.standard_normal((half, P))
    # build_population would leave the sigma==0 columns identical to the
    # mean (eps*0 == 0); mimic that here directly on eps_full instead of
    # going through build_population, since es_gradient only ever sees
    # eps_half/fitness/sigma.
    fitness = rng.standard_normal(2 * half)

    grad = es_gradient(eps_half, fitness, sigma)
    assert grad.shape == (P,)
    assert np.all(np.isfinite(grad)), "zero-sigma entries must not produce NaN/inf gradients"
    np.testing.assert_array_equal(grad[sigma == 0.0], 0.0)


def test_es_gradient_all_zero_sigma_gives_all_zero_grad():
    rng = np.random.default_rng(1)
    P = 4
    half = 4
    eps_half = rng.standard_normal((half, P))
    fitness = rng.standard_normal(2 * half)
    grad = es_gradient(eps_half, fitness, 0.0)
    np.testing.assert_array_equal(grad, np.zeros(P))


def test_es_gradient_scalar_sigma_matches_previous_behaviour():
    """Regression: nonzero scalar sigma must give the same result as before
    the zero-sigma fix (plain elementwise divide)."""
    rng = np.random.default_rng(2)
    P = 5
    half = 6
    eps_half = rng.standard_normal((half, P))
    fitness = rng.standard_normal(2 * half)
    sigma = 0.2
    grad = es_gradient(eps_half, fitness, sigma)

    eps_full = np.concatenate([eps_half, -eps_half], axis=0)
    order = np.argsort(fitness)
    ranks = np.empty(len(fitness))
    ranks[order] = np.arange(len(fitness))
    centered = ranks / (len(fitness) - 1) - 0.5
    expected = (eps_full * centered[:, None]).sum(axis=0) / (2 * half * sigma)
    np.testing.assert_allclose(grad, expected)


def test_build_population_zero_sigma_freezes_those_params():
    rng = np.random.default_rng(3)
    mean_theta = np.arange(N_PARAMS, dtype=np.float64)
    sigma_vec = steering_param_sigma_vector(0.05, 0.3, N_PARAMS)
    sigma_vec[N_ENC_PARAMS:] = 0.0  # freeze the decoder block, like --freeze-decoder
    theta_pop, _eps = build_population(mean_theta, sigma_vec, 4, rng)
    # every population member's decoder block must equal the mean EXACTLY
    for row in theta_pop:
        np.testing.assert_array_equal(row[N_ENC_PARAMS:], mean_theta[N_ENC_PARAMS:])
    # encoder block should generally NOT all be identical to the mean
    assert not np.allclose(theta_pop[:, :N_ENC_PARAMS], mean_theta[:N_ENC_PARAMS])


# ---------------------------------------------------------------------
# Task D(iii): averaging fitness/welfare/metrics/DAN rates across several
# common-random-number env seeds within one generation
# ---------------------------------------------------------------------

def _fake_episode_result(seed_offset: float, B: int = 3) -> EpisodeResult:
    fitness = np.arange(B, dtype=np.float64) + seed_offset
    welfare = fitness * 2.0
    metrics = [{k: float(i + seed_offset) for k in METRIC_KEYS} for i in range(B)]
    dan = np.full(B, seed_offset)
    dan_driven = np.full(B, seed_offset * 2)
    return EpisodeResult(fitness=fitness, welfare=welfare, metrics=metrics,
                          mean_dan_rate=dan, mean_dan_rate_driven=dan_driven)


def test_average_episode_results_averages_elementwise():
    r1 = _fake_episode_result(0.0)
    r2 = _fake_episode_result(10.0)
    avg = _average_episode_results([r1, r2])
    np.testing.assert_allclose(avg.fitness, (r1.fitness + r2.fitness) / 2.0)
    np.testing.assert_allclose(avg.welfare, (r1.welfare + r2.welfare) / 2.0)
    np.testing.assert_allclose(avg.mean_dan_rate, (r1.mean_dan_rate + r2.mean_dan_rate) / 2.0)
    np.testing.assert_allclose(avg.mean_dan_rate_driven,
                                (r1.mean_dan_rate_driven + r2.mean_dan_rate_driven) / 2.0)
    assert len(avg.metrics) == len(r1.metrics)
    for i in range(len(avg.metrics)):
        for k in METRIC_KEYS:
            assert avg.metrics[i][k] == pytest.approx((r1.metrics[i][k] + r2.metrics[i][k]) / 2.0)


def test_average_episode_results_single_result_is_identity():
    r1 = _fake_episode_result(5.0)
    avg = _average_episode_results([r1])
    np.testing.assert_allclose(avg.fitness, r1.fitness)
    np.testing.assert_allclose(avg.welfare, r1.welfare)


# ---------------------------------------------------------------------
# Task D end-to-end: --freeze-decoder via the real CLI (main()), tiny
# population/episode so it runs in a few seconds beyond BrainPolicy's own
# one-time brain-weight load.
# ---------------------------------------------------------------------

def test_main_gens_zero_writes_expected_defaults_to_config(tmp_path, monkeypatch):
    """gens=0 skips the ES loop entirely (still constructs BrainPolicy, but
    no rollout) -- cheap way to check the new Task D defaults actually land
    in config.json via the real argparse wiring."""
    monkeypatch.setattr(train_es, "RESULTS_ROOT", tmp_path)
    train_es.main([
        "--mode", "welfare", "--run-name", "gens0test", "--gens", "0",
        "--population", "4", "--eval-every", "0", "--threads", "2",
    ])
    config = json.loads((tmp_path / "gens0test" / "config.json").read_text())
    assert config["gamma"] == pytest.approx(1.0)
    assert config["eval_episodes"] == 16
    assert config["freeze_decoder"] is True
    assert config["seeds_per_gen"] == 2
    assert config["contrast_weighting"] == "sqrt"

    ckpt = np.load(tmp_path / "gens0test" / "ckpt.npz")
    np.testing.assert_array_equal(ckpt["mean_theta"], default_params(seed=config["seed"]).astype(np.float32))


def test_main_freeze_decoder_keeps_decoder_exactly_fixed_over_real_generations(tmp_path, monkeypatch):
    monkeypatch.setattr(train_es, "RESULTS_ROOT", tmp_path)
    init_theta = default_params(seed=0).astype(np.float32)

    train_es.main([
        "--mode", "welfare", "--run-name", "freezetest", "--gens", "2",
        "--population", "4", "--n-steps", "5", "--steps-per-action", "2",
        "--eval-every", "0", "--threads", "2", "--seeds-per-gen", "1",
        "--freeze-decoder",
    ])
    ckpt = np.load(tmp_path / "freezetest" / "ckpt.npz")
    final_theta = ckpt["mean_theta"]

    np.testing.assert_array_equal(
        final_theta[N_ENC_PARAMS:], init_theta[N_ENC_PARAMS:],
        err_msg="the decoder block must be BIT-IDENTICAL to its init after --freeze-decoder training",
    )
    assert not np.allclose(final_theta[:N_ENC_PARAMS], init_theta[:N_ENC_PARAMS]), (
        "the encoder block should have moved from its init over 2 real generations"
    )


def test_main_no_freeze_decoder_lets_decoder_move(tmp_path, monkeypatch):
    monkeypatch.setattr(train_es, "RESULTS_ROOT", tmp_path)
    init_theta = default_params(seed=0).astype(np.float32)

    train_es.main([
        "--mode", "welfare", "--run-name", "nofreezetest", "--gens", "2",
        "--population", "4", "--n-steps", "5", "--steps-per-action", "2",
        "--eval-every", "0", "--threads", "2", "--seeds-per-gen", "1",
        "--no-freeze-decoder",
    ])
    ckpt = np.load(tmp_path / "nofreezetest" / "ckpt.npz")
    final_theta = ckpt["mean_theta"]
    assert not np.allclose(final_theta[N_ENC_PARAMS:], init_theta[N_ENC_PARAMS:]), (
        "--no-freeze-decoder should let the decoder move"
    )
