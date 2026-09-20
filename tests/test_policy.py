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
    N_GROUPS, N_POOLS, N_FEATURES, N_ACTIONS, N_ENC_PARAMS, N_STEER_PARAMS,
    steering_param_names, unpack_steering_params, steering_param_sigma_vector,
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
# Lateralized sensory drive (v4: shared steer_L/steer_R pathway)
# ---------------------------------------------------------------------

def test_lateralized_group_rates_food_left_vs_right():
    """v4: food drives the SAME shared steer_L/steer_R pair smoke/reels do
    (GROUP_NAMES[0]/[1]) -- there is no more dedicated food_odor group."""
    B = 2
    pol = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=0)
    theta = np.tile(default_params(), (B, 1)).astype(np.float32)
    pol.set_params(theta)
    assert GROUP_NAMES[0] == "steer_L" and GROUP_NAMES[1] == "steer_R"

    obs = np.zeros((B, 12), dtype=np.float32)
    obs[0, 0] = 1.0  # food_odor_L = 1, food on the left
    obs[1, 1] = 1.0  # food_odor_R = 1, food on the right

    rates = pol.group_rates(obs)  # (B, 8): [steer_L, steer_R, ...]
    steer_L_rate = rates[:, 0]
    steer_R_rate = rates[:, 1]

    assert steer_L_rate[0] > steer_R_rate[0], "food on the left should drive the shared steer_L group harder"
    assert steer_R_rate[1] > steer_L_rate[1], "food on the right should drive the shared steer_R group harder"
    assert not np.isclose(steer_L_rate[0], steer_L_rate[1]), "L group rate should differ between the two conditions"


# ---------------------------------------------------------------------
# v4 encoder: mirror symmetry, sign of c, per-batch independence
# ---------------------------------------------------------------------

def test_encoder_mirror_symmetry_full_lr_swap():
    """Swapping EVERY source's L/R obs values must swap steer_L/steer_R
    rates EXACTLY (same a/c/m/h/b weights are shared by both sides -- no
    separate 'left encoder'/'right encoder'), for an arbitrary (nonzero
    m, h) parameter vector, not just the all-zero default init."""
    B = 1
    pol = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=0)
    theta = default_params(seed=0).copy()
    rng = np.random.default_rng(3)
    # perturb m and h away from their all-zero default so the test isn't
    # vacuous (m/h terms would trivially vanish otherwise).
    theta[:N_STEER_PARAMS] = default_params(seed=0)[:N_STEER_PARAMS]
    names = steering_param_names()
    for i, name in enumerate(names):
        if name.startswith("m_") or name.startswith("h_"):
            theta[i] = rng.uniform(-1.0, 1.0)
    theta = np.tile(theta, (B, 1)).astype(np.float32)
    pol.set_params(theta)

    obs = rng.uniform(0.05, 0.95, size=(B, 12)).astype(np.float32)
    obs[:, 6:] = rng.uniform(0.0, 1.0, size=(B, 6))  # contact/state channels, irrelevant to swap

    rates = pol.group_rates(obs)
    obs_swapped = obs.copy()
    for li, ri in [(0, 1), (2, 3), (4, 5)]:
        obs_swapped[:, [li, ri]] = obs_swapped[:, [ri, li]]
    rates_swapped = pol.group_rates(obs_swapped)

    np.testing.assert_allclose(rates_swapped[:, 0], rates[:, 1], atol=1e-5,
                                err_msg="swapped steer_L rate should equal ORIGINAL steer_R rate")
    np.testing.assert_allclose(rates_swapped[:, 1], rates[:, 0], atol=1e-5,
                                err_msg="swapped steer_R rate should equal ORIGINAL steer_L rate")
    # contact/intero groups (indices 2..7) never touch L/R obs -- unaffected by the swap.
    np.testing.assert_allclose(rates_swapped[:, 2:], rates[:, 2:], atol=1e-6)


def test_negative_c_steers_away_from_source():
    """A NEGATIVE c_src makes the shared pathway respond MORE on the side
    OPPOSITE a source on X's own side (steers away), while a POSITIVE
    c_src (the default init sign) responds more on the source's own side
    (steers toward it)."""
    B = 2
    pol = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=0)
    theta = default_params(seed=0)
    names = steering_param_names()
    c_food_idx = names.index("c_food")

    theta_pos = theta.copy()
    theta_pos[c_food_idx] = 3.0    # default sign: attraction
    theta_neg = theta.copy()
    theta_neg[c_food_idx] = -3.0   # flipped sign: aversion

    batched = np.stack([theta_pos, theta_neg], axis=0).astype(np.float32)
    pol.set_params(batched)

    obs = np.zeros((B, 12), dtype=np.float32)
    obs[:, 0] = 0.8  # food_odor_L: food on the LEFT for both rows
    obs[:, 1] = 0.1

    rates = pol.group_rates(obs)
    assert rates[0, 0] > rates[0, 1], "c_food > 0: food on the left should drive steer_L harder (toward)"
    assert rates[1, 1] > rates[1, 0], "c_food < 0: food on the left should drive steer_R harder (away)"


def test_steering_params_per_batch_element_independence():
    """Perturbing ONE batch row's steering params (not just any param) must
    only change that row's ENCODER OUTPUT (group_rates), matching the
    general batch-independence property but specifically exercised on the
    v4 steering block. Uses group_rates (pure function of obs+params, no
    brain noise) rather than full act() so a moderate, realistic parameter
    delta is guaranteed to be visible instead of possibly being swallowed
    by the frozen brain's own Poisson stochasticity."""
    B = 3
    obs = np.zeros((B, 12), dtype=np.float32)
    obs[:, 2] = 0.9  # smoke_odor_L: strong left contrast for all 3 rows
    obs[:, 3] = 0.1  # smoke_odor_R

    theta_base = np.tile(default_params(seed=0), (B, 1)).astype(np.float32)
    theta_perturbed = theta_base.copy()
    names = steering_param_names()
    theta_perturbed[1, names.index("c_smoke")] += 5.0  # row 1 only

    pol = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=11)
    pol.set_params(theta_base)
    rates_base = pol.group_rates(obs)
    pol.set_params(theta_perturbed)
    rates_perturbed = pol.group_rates(obs)

    np.testing.assert_array_equal(rates_base[0], rates_perturbed[0])
    np.testing.assert_array_equal(rates_base[2], rates_perturbed[2])
    assert not np.allclose(rates_base[1], rates_perturbed[1]), (
        "perturbing row 1's c_smoke had no effect on row 1's steer_L/steer_R rates"
    )


# ---------------------------------------------------------------------
# DAN logging split (driven input DANs vs other)
# ---------------------------------------------------------------------

def test_dan_rate_split_shapes_and_nonnegative():
    B = 2
    pol = BrainPolicy(batch=B, device="cpu", dt=0.5, steps_per_action=4, seed=0)
    pol.reset()
    obs = np.random.default_rng(0).uniform(0, 1, size=(B, 12)).astype(np.float32)
    pol.act(obs)
    other = pol.mean_dan_rate_hz()
    driven = pol.dan_rate_driven_hz()
    assert other.shape == (B,) and driven.shape == (B,)
    assert np.all(np.isfinite(other)) and np.all(np.isfinite(driven))
    assert np.all(other >= 0.0) and np.all(driven >= 0.0)
    assert pol.n_dan_driven + pol.n_dan_other == pol.n_dan


# ---------------------------------------------------------------------
# Per-parameter ES sigma (Task 3)
# ---------------------------------------------------------------------

def test_steering_param_sigma_vector_layout():
    vec = steering_param_sigma_vector(0.05, 0.3, N_PARAMS)
    assert vec.shape == (N_PARAMS,)
    np.testing.assert_allclose(vec[:N_STEER_PARAMS], 0.3)
    np.testing.assert_allclose(vec[N_STEER_PARAMS:], 0.05)


def test_build_population_and_es_gradient_accept_per_parameter_sigma():
    rng = np.random.default_rng(0)
    mean_theta = default_params(seed=0).astype(np.float64)
    sigma_vec = steering_param_sigma_vector(0.05, 0.3, mean_theta.shape[0])
    half = 4
    theta_pop, eps_half = build_population(mean_theta, sigma_vec, half, rng)
    assert theta_pop.shape == (2 * half, N_PARAMS)
    # steering-block spread should be MUCH larger than decoder-block spread,
    # since sigma_steer=0.3 >> sigma=0.05.
    steer_spread = float(np.std(theta_pop[:, :N_STEER_PARAMS] - mean_theta[:N_STEER_PARAMS]))
    dec_spread = float(np.std(theta_pop[:, N_STEER_PARAMS:] - mean_theta[N_STEER_PARAMS:]))
    assert steer_spread > dec_spread

    fitness = rng.standard_normal(2 * half)
    grad = es_gradient(eps_half, fitness, sigma_vec)
    assert grad.shape == (N_PARAMS,)
    assert np.all(np.isfinite(grad))


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
# v2/v4 (Task B / Task 1): per-group max rate, anatomical-prior decoder
# init, z-normalized readout cache (fast AND slow trace stats), and the
# empirical turn-sign convention the prior init depends on.
# ---------------------------------------------------------------------

def test_readout_cache_v4_shapes():
    (readout_idx, pool_assign, dan_idx, n_pools, pool_mean, pool_std, pool_turn_sign,
     pool_mean_slow, pool_std_slow) = _load_readout_cache()
    assert n_pools == 64
    assert pool_mean.shape == (64,)
    assert pool_std.shape == (64,)
    assert pool_turn_sign.shape == (64,)
    assert np.all(pool_std > 0), "z-norm std must be floored above 0"
    assert set(np.unique(pool_turn_sign).tolist()) <= {-1.0, 1.0}
    assert pool_assign.shape == readout_idx.shape
    assert pool_assign.min() >= 0 and pool_assign.max() < 64
    # Task 1: fixed z-norm stats for the optional tau=200ms slow-trace feature.
    assert pool_mean_slow is not None and pool_std_slow is not None
    assert pool_mean_slow.shape == (64,)
    assert pool_std_slow.shape == (64,)
    assert np.all(pool_std_slow > 0)


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
    (_, _, _, _, _, _, pool_turn_sign, _, _) = _load_readout_cache()
    theta = default_params(seed=0)
    off = N_ENC_PARAMS
    W_dec = theta[off:off + N_ACTIONS * N_FEATURES].reshape(N_ACTIONS, N_FEATURES)
    b_dec = theta[off + N_ACTIONS * N_FEATURES:]
    assert theta.shape == (N_PARAMS,)

    # turn output's weight on the first 64 (tau=50ms) pool features always
    # gets the anatomical-prior sign, regardless of USE_SLOW_TRACE/
    # ADD_NO_SIGNAL_FEATURE (which only ever APPEND more feature columns).
    np.testing.assert_allclose(W_dec[0, :N_POOLS], DEFAULT_INIT_SCALE * pool_turn_sign, atol=1e-6)
    assert b_dec[1] == pytest.approx(0.5), "forward bias should default to +0.5"

    theta_scaled = default_params(seed=0, init_scale=1.5)
    W_dec_scaled = theta_scaled[off:off + N_ACTIONS * N_FEATURES].reshape(N_ACTIONS, N_FEATURES)
    np.testing.assert_allclose(W_dec_scaled[0, :N_POOLS], 1.5 * pool_turn_sign, atol=1e-6)


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
