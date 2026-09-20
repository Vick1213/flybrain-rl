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

from flyrl.policy import N_PARAMS, N_STEER_PARAMS, steering_param_names, steering_param_sigma_vector
from flyrl.train_es import (
    Adam, build_population, es_gradient, clip_steering_params,
    STEER_CLIP_B, STEER_CLIP_ACMH,
)


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
