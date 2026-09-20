"""Fast tests for flyrl.policy.BrainPolicy and one end-to-end ES generation
via flyrl.train_es. Designed to run in well under 2 minutes total.

Covers: parameter/action shapes, determinism given seeds, batch
independence (params of row i only affect action i), lateralized sensory
drive from an untrained policy (food on the left vs right), and one full ES
generation with a tiny population/episode length.
"""

from __future__ import annotations

import numpy as np
import pytest

from flyrl.policy import (
    BrainPolicy, default_params, N_PARAMS, _load_readout_cache, DEFAULT_INIT_SCALE,
    N_GROUPS, N_ENC_IN, N_POOLS, N_ACTIONS,
)
from flyrl.io_neurons import GROUP_NAMES, GROUP_MAX_RATE_HZ
from flyrl.addiction_env import FlyAddictionEnv, VecFlyAddictionEnv
from flyrl.train_es import run_episode_batch, build_population, es_gradient, Adam
from flyrl.taxis import run_taxis_generation, mask_obs, MODALITIES, MODALITY_KEEP_CHANNELS


# ---------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------

def test_shapes():
    B = 3
    pol = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=0)
    assert pol.n_params == N_PARAMS
    theta = np.tile(default_params(), (B, 1)).astype(np.float32)
    pol.set_params(theta)
    pol.reset()
    obs = np.random.default_rng(0).uniform(0, 1, size=(B, 12)).astype(np.float32)
    actions = pol.act(obs)
    assert actions.shape == (B, 2)
    assert np.all(np.isfinite(actions))
    assert np.all(actions >= -1.0) and np.all(actions <= 1.0)
    dan = pol.mean_dan_rate_hz()
    assert dan.shape == (B,)
    assert np.all(dan >= 0.0)


# ---------------------------------------------------------------------
# Determinism given seeds
# ---------------------------------------------------------------------

def test_determinism_same_seed():
    B = 2
    theta = np.tile(default_params(), (B, 1)).astype(np.float32)
    obs = np.random.default_rng(1).uniform(0, 1, size=(B, 12)).astype(np.float32)

    pol_a = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=42)
    pol_a.set_params(theta)
    pol_a.reset()
    act_a = pol_a.act(obs)

    pol_b = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=42)
    pol_b.set_params(theta)
    pol_b.reset()
    act_b = pol_b.act(obs)

    np.testing.assert_array_equal(act_a, act_b)


# ---------------------------------------------------------------------
# Batch independence: params of row i only affect action i
# ---------------------------------------------------------------------

def test_batch_independence_of_params():
    B = 3
    obs = np.random.default_rng(2).uniform(0, 1, size=(B, 12)).astype(np.float32)

    theta_base = np.tile(default_params(), (B, 1)).astype(np.float32)
    theta_perturbed = theta_base.copy()
    theta_perturbed[1] += 5.0  # large perturbation, row 1 only

    pol_a = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=7)
    pol_a.set_params(theta_base)
    pol_a.reset()
    act_a = pol_a.act(obs)

    pol_b = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=7)
    pol_b.set_params(theta_perturbed)
    pol_b.reset()
    act_b = pol_b.act(obs)

    # Rows 0 and 2 (unperturbed params, same obs, same brain seed) must be
    # bit-identical between the two runs; row 1 (perturbed) must differ.
    np.testing.assert_array_equal(act_a[0], act_b[0])
    np.testing.assert_array_equal(act_a[2], act_b[2])
    assert not np.allclose(act_a[1], act_b[1]), "perturbing row 1's params had no effect on action 1"


# ---------------------------------------------------------------------
# Lateralized sensory drive
# ---------------------------------------------------------------------

def test_lateralized_group_rates_food_left_vs_right():
    B = 2
    pol = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=0)
    theta = np.tile(default_params(), (B, 1)).astype(np.float32)
    pol.set_params(theta)

    obs = np.zeros((B, 12), dtype=np.float32)
    obs[0, 0] = 1.0  # food_odor_L = 1, food on the left
    obs[0, 1] = 0.0
    obs[1, 0] = 0.0
    obs[1, 1] = 1.0  # food_odor_R = 1, food on the right

    rates = pol.group_rates(obs)  # (B, 12): [..., food_odor_L=0, food_odor_R=1, ...]
    food_L_rate = rates[:, 0]
    food_R_rate = rates[:, 1]

    assert food_L_rate[0] > food_R_rate[0], "food on the left should drive food_odor_L group harder"
    assert food_R_rate[1] > food_L_rate[1], "food on the right should drive food_odor_R group harder"
    assert not np.isclose(food_L_rate[0], food_L_rate[1]), "L group rate should differ between the two conditions"


# ---------------------------------------------------------------------
# One ES generation end-to-end (tiny population + short episode)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["welfare", "hijacked"])
def test_one_es_generation_end_to_end(mode):
    B = 4
    n_steps = 20
    sigma = 0.1

    policy = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=0)
    vec_env = VecFlyAddictionEnv(num_envs=B, n_steps=n_steps)

    mean_theta = default_params(seed=0).astype(np.float64)
    rng = np.random.default_rng(0)
    theta_pop, eps_half = build_population(mean_theta, sigma, B // 2, rng)
    assert theta_pop.shape == (B, N_PARAMS)

    result = run_episode_batch(policy, vec_env, theta_pop, seed=123, n_steps=n_steps, mode=mode)
    assert result.fitness.shape == (B,)
    assert result.welfare.shape == (B,)
    assert len(result.metrics) == B
    assert result.mean_dan_rate.shape == (B,)
    assert np.all(np.isfinite(result.fitness))
    assert np.all(np.isfinite(result.welfare))

    grad = es_gradient(eps_half, result.fitness, sigma)
    assert grad.shape == (N_PARAMS,)
    assert np.all(np.isfinite(grad))

    adam = Adam(N_PARAMS, lr=0.03)
    update = adam.step(grad)
    new_mean = mean_theta + update
    assert new_mean.shape == (N_PARAMS,)
    assert np.all(np.isfinite(new_mean))
    assert not np.allclose(new_mean, mean_theta), "ES update should move the mean parameters"


# ---------------------------------------------------------------------
# v2 (Task B): per-group max rate, anatomical-prior decoder init,
# z-normalized readout cache, and the empirical turn-sign convention the
# prior init depends on.
# ---------------------------------------------------------------------

def test_readout_cache_v2_shapes():
    readout_idx, pool_assign, dan_idx, n_pools, pool_mean, pool_std, pool_turn_sign = _load_readout_cache()
    assert n_pools == 64
    assert pool_mean.shape == (64,)
    assert pool_std.shape == (64,)
    assert pool_turn_sign.shape == (64,)
    assert np.all(pool_std > 0), "z-norm std must be floored above 0"
    assert set(np.unique(pool_turn_sign).tolist()) <= {-1.0, 1.0}
    assert pool_assign.shape == readout_idx.shape
    assert pool_assign.min() >= 0 and pool_assign.max() < 64


def test_group_max_rate_matches_io_neurons_and_is_respected():
    B = 2
    pol = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=0)
    expected = np.array([GROUP_MAX_RATE_HZ[name] for name in GROUP_NAMES], dtype=np.float32)
    np.testing.assert_array_equal(pol.group_max_rate.cpu().numpy(), expected)

    theta = np.tile(default_params(), (B, 1)).astype(np.float32)
    pol.set_params(theta)
    # Drive every obs channel maximally; sigmoid saturates towards 1, so
    # each group's rate should approach (but never exceed) its own ceiling.
    obs = np.ones((B, 12), dtype=np.float32)
    rates = pol.group_rates(obs)
    for gi, name in enumerate(GROUP_NAMES):
        assert rates[:, gi].max() <= GROUP_MAX_RATE_HZ[name] + 1e-3, (
            f"group {name} exceeded its configured max rate {GROUP_MAX_RATE_HZ[name]}"
        )


def test_anatomical_prior_decoder_init():
    _, _, _, _, _, _, pool_turn_sign = _load_readout_cache()
    theta = default_params(seed=0)
    off = N_GROUPS * N_ENC_IN + N_GROUPS
    W_dec = theta[off:off + N_ACTIONS * N_POOLS].reshape(N_ACTIONS, N_POOLS)
    b_dec = theta[off + N_ACTIONS * N_POOLS:]

    np.testing.assert_allclose(W_dec[0, :], DEFAULT_INIT_SCALE * pool_turn_sign, atol=1e-6)
    assert b_dec[1] == pytest.approx(0.5), "forward bias should default to +0.5"

    theta_scaled = default_params(seed=0, init_scale=1.5)
    W_dec_scaled = theta_scaled[off:off + N_ACTIONS * N_POOLS].reshape(N_ACTIONS, N_POOLS)
    np.testing.assert_allclose(W_dec_scaled[0, :], 1.5 * pool_turn_sign, atol=1e-6)


def test_turn_sign_convention_empirical():
    """Locks in the env's turn-sign convention that the anatomical-prior
    decoder init (default_params) relies on: action[0] > 0 turns the fly
    TOWARD its own left (the sensor at heading + pi/2), matching the
    bilateral klinotaxis sign used by flyrl.scripted._steer_towards
    (turn = turn_gain * (L - R))."""
    env = FlyAddictionEnv()
    env.reset(seed=0)
    env.pos = np.array([0.5, 0.5])
    env.theta = 0.0
    env.sources["food"] = np.array([0.5, 0.7])  # placed at bearing +pi/2 (fly's left)

    def bearing_to_food():
        d = env.sources["food"] - env.pos
        return (np.arctan2(d[1], d[0]) - env.theta + np.pi) % (2 * np.pi) - np.pi

    bearing_before = bearing_to_food()
    assert bearing_before == pytest.approx(np.pi / 2, abs=1e-6)

    env.step(np.array([1.0, -1.0]))  # max positive turn action, no forward motion
    bearing_after = bearing_to_food()
    assert bearing_after < bearing_before, (
        "action[0] > 0 should turn the fly toward a source on its left "
        "(decreasing the bearing toward 0), matching default_params' "
        "anatomical-prior sign convention"
    )


# ---------------------------------------------------------------------
# Task C: taxis pretraining (flyrl.taxis / train_es --mode taxis)
# ---------------------------------------------------------------------

def test_mask_obs_keeps_only_target_modality_and_interoceptive():
    obs = np.arange(12, dtype=np.float32).reshape(1, 12) + 1.0  # all channels nonzero
    for modality in MODALITIES:
        masked = mask_obs(obs, modality)
        keep = MODALITY_KEEP_CHANNELS[modality]
        for ch in range(12):
            if ch in keep:
                assert masked[0, ch] == obs[0, ch]
            else:
                assert masked[0, ch] == 0.0


def test_run_taxis_generation_shapes_and_dense_fitness():
    B = 4
    policy = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=0)
    vec_env = VecFlyAddictionEnv(num_envs=B, n_steps=300)

    mean_theta = default_params(seed=0).astype(np.float64)
    rng = np.random.default_rng(0)
    theta_pop, _eps_half = build_population(mean_theta, 0.1, B // 2, rng)

    result = run_taxis_generation(policy, vec_env, theta_pop, seed=123, n_substeps=10)
    assert result.fitness.shape == (B,)
    assert np.all(np.isfinite(result.fitness))
    assert set(result.reach_rate.keys()) == set(MODALITIES)
    for modality in MODALITIES:
        assert result.reach_rate[modality].shape == (B,)
        assert result.reach_rate[modality].dtype == bool
        assert result.dist_reduction[modality].shape == (B,)
        assert np.all(np.isfinite(result.dist_reduction[modality]))


def test_taxis_common_random_numbers_across_population():
    """Common random numbers: every population member must see the IDENTICAL
    environment realization (source positions + fly start pose) for a given
    modality's sub-episode within a generation -- i.e. fitness differences
    come from parameter differences (and the brain's own independent
    Poisson noise per batch row -- individual flies are not deterministic
    copies of each other even given identical params/obs), not from
    different odor/light source layouts."""
    from flyrl.taxis import _vec_reset_common_seed

    B = 4
    vec_env = VecFlyAddictionEnv(num_envs=B, n_steps=300)
    for mi, modality in enumerate(MODALITIES):
        mod_seed = 123 + mi * 1_000_003
        obs = _vec_reset_common_seed(vec_env, mod_seed)
        np.testing.assert_allclose(obs, np.broadcast_to(obs[0], obs.shape), atol=1e-12), (
            f"all population members should see identical initial obs for {modality} (CRN)"
        )
        source_positions = np.stack([env.sources[modality] for env in vec_env.envs])
        np.testing.assert_allclose(source_positions, np.broadcast_to(source_positions[0], source_positions.shape),
                                    atol=1e-12), (
            f"all population members should share the same {modality} source position (CRN)"
        )
