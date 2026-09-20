"""Tests for flyrl.addiction_env.FlyAddictionEnv.

Covers: gymnasium.utils.env_checker.check_env compliance, seeding
determinism, observation shape/bounds, and the scripted-policy behavioural
properties the reward tuning is supposed to produce (see the module
docstring in flyrl/addiction_env.py and flyrl/scripted.py for the intended
addiction narrative).
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from flyrl.addiction_env import (
    FlyAddictionEnv,
    VecFlyAddictionEnv,
    AddictionParams,
    addiction_metrics,
)
from flyrl.scripted import (
    ForagerPolicy,
    SmokerPolicy,
    ReelsPolicy,
    GreedyPolicy,
    run_episode,
)

N_SEEDS = 20


# ---------------------------------------------------------------------
# Basic Gymnasium API compliance
# ---------------------------------------------------------------------

def test_check_env_no_render():
    env = FlyAddictionEnv()
    check_env(env, skip_render_check=True)


def test_check_env_with_render():
    env = FlyAddictionEnv(render_mode="rgb_array")
    check_env(env, skip_render_check=False)


def test_obs_space_shape_and_bounds():
    env = FlyAddictionEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (12,)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    assert len(FlyAddictionEnv.obs_channels) == 12

    rng = np.random.default_rng(0)
    for _ in range(50):
        action = rng.uniform(-1, 1, size=(2,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (12,)
        assert obs.dtype == np.float32
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        assert isinstance(reward, float)
        assert "at" in info and info["at"] in (None, "food", "smoke", "reels")
        for key in ("h", "n", "tau", "w"):
            assert key in info


def test_episode_length_default():
    env = FlyAddictionEnv()
    env.reset(seed=0)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        steps += 1
    assert steps == 300
    assert not terminated
    assert truncated


def test_episode_length_configurable():
    env = FlyAddictionEnv(n_steps=50)
    env.reset(seed=0)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        steps += 1
    assert steps == 50


# ---------------------------------------------------------------------
# Seeding determinism
# ---------------------------------------------------------------------

def test_reset_determinism_same_seed():
    env_a = FlyAddictionEnv()
    env_b = FlyAddictionEnv()
    obs_a, info_a = env_a.reset(seed=42)
    obs_b, info_b = env_b.reset(seed=42)
    np.testing.assert_array_equal(obs_a, obs_b)
    for name in ("food", "smoke", "reels"):
        np.testing.assert_array_equal(env_a.sources[name], env_b.sources[name])
    np.testing.assert_array_equal(env_a.pos, env_b.pos)
    assert env_a.theta == env_b.theta


def test_step_trajectory_determinism_same_seed():
    rng_actions = np.random.default_rng(7)
    actions = [rng_actions.uniform(-1, 1, size=2).astype(np.float32) for _ in range(40)]

    def rollout(seed):
        env = FlyAddictionEnv()
        obs, _ = env.reset(seed=seed)
        obs_hist = [obs]
        reward_hist = []
        for a in actions:
            obs, r, term, trunc, info = env.step(a)
            obs_hist.append(obs)
            reward_hist.append(r)
        return obs_hist, reward_hist

    obs_hist_a, reward_hist_a = rollout(123)
    obs_hist_b, reward_hist_b = rollout(123)
    for oa, ob in zip(obs_hist_a, obs_hist_b):
        np.testing.assert_array_equal(oa, ob)
    assert reward_hist_a == reward_hist_b


def test_different_seeds_give_different_sources():
    env = FlyAddictionEnv()
    env.reset(seed=1)
    sources_1 = {k: v.copy() for k, v in env.sources.items()}
    env.reset(seed=2)
    sources_2 = env.sources
    assert not all(np.allclose(sources_1[k], sources_2[k]) for k in sources_1)


# ---------------------------------------------------------------------
# Scripted-policy behavioural properties
# ---------------------------------------------------------------------

def _returns_and_metrics(policy_factory, n_seeds=N_SEEDS, params=None):
    returns = []
    metrics_list = []
    infos_list = []
    for seed in range(n_seeds):
        env = FlyAddictionEnv(params=params)
        policy = policy_factory()
        total_reward, infos = run_episode(env, policy, seed=seed)
        returns.append(total_reward)
        metrics_list.append(addiction_metrics(infos))
        infos_list.append(infos)
    return np.array(returns), metrics_list, infos_list


def test_forager_beats_reels_and_smoker():
    forager_returns, _, _ = _returns_and_metrics(lambda: ForagerPolicy())
    reels_returns, _, _ = _returns_and_metrics(lambda: ReelsPolicy())
    smoker_returns, _, _ = _returns_and_metrics(lambda: SmokerPolicy())

    assert forager_returns.mean() > reels_returns.mean()
    assert forager_returns.mean() > smoker_returns.mean()
    # forager should be a "sensible" policy with solidly positive return
    assert forager_returns.mean() > 0


def test_smoker_opponent_process():
    _, metrics_list, infos_list = _returns_and_metrics(lambda: SmokerPolicy())

    final_tolerances = np.array([m["final_tolerance"] for m in metrics_list])
    assert final_tolerances.mean() > 0.5

    first_third_means = []
    last_third_means = []
    for infos in infos_list:
        n = len(infos)
        third = max(1, n // 3)
        rewards = np.array([info["reward"] for info in infos])
        first_third_means.append(rewards[:third].mean())
        last_third_means.append(rewards[-third:].mean())
    assert np.mean(last_third_means) < np.mean(first_third_means)


def test_forager_steering_reaches_food():
    n_seeds = N_SEEDS
    reached = 0
    for seed in range(n_seeds):
        env = FlyAddictionEnv()
        policy = ForagerPolicy()
        _, infos = run_episode(env, policy, seed=seed)
        if any(info["at"] == "food" for info in infos):
            reached += 1
    assert reached / n_seeds > 0.9


def test_greedy_policy_runs():
    # GreedyPolicy has no hard numeric assertion in the spec beyond existing
    # and being usable; just make sure it runs cleanly end to end and its
    # `update` hook is exercised.
    returns, metrics_list, _ = _returns_and_metrics(lambda: GreedyPolicy(seed=0), n_seeds=5)
    assert returns.shape == (5,)
    for m in metrics_list:
        assert 0.0 <= m["final_tolerance"] <= 1.0


# ---------------------------------------------------------------------
# addiction_metrics helper
# ---------------------------------------------------------------------

def test_addiction_metrics_empty():
    m = addiction_metrics([])
    assert m["frac_none"] == 1.0
    assert m["time_to_first_smoke"] is None


def test_addiction_metrics_time_to_first_smoke():
    env = FlyAddictionEnv()
    policy = SmokerPolicy()
    _, infos = run_episode(env, policy, seed=0)
    m = addiction_metrics(infos)
    assert m["time_to_first_smoke"] is not None
    assert 0 <= m["time_to_first_smoke"] < len(infos)
    assert 0.0 <= m["compulsion_smoke"] <= 1.0
    assert 0.0 <= m["frac_smoke"] <= 1.0


# ---------------------------------------------------------------------
# VecFlyAddictionEnv
# ---------------------------------------------------------------------

def test_vec_env_reset_and_step_shapes():
    B = 4
    venv = VecFlyAddictionEnv(B)
    obs = venv.reset(seed=0)
    assert obs.shape == (B, 12)

    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(B, 2)).astype(np.float32)
    obs, rewards, dones, infos = venv.step(actions)
    assert obs.shape == (B, 12)
    assert rewards.shape == (B,)
    assert dones.shape == (B,)
    assert len(infos) == B
    assert not dones.any()


def test_vec_env_no_autoreset_stays_done():
    B = 2
    venv = VecFlyAddictionEnv(B, params=AddictionParams(n_steps=3))
    venv.reset(seed=0)
    actions = np.zeros((B, 2), dtype=np.float32)
    for _ in range(3):
        obs, rewards, dones, infos = venv.step(actions)
    assert dones.all()
    last_obs = obs.copy()

    # stepping again after done: zero reward, still done, obs unchanged
    obs2, rewards2, dones2, infos2 = venv.step(actions)
    np.testing.assert_array_equal(obs2, last_obs)
    assert np.all(rewards2 == 0.0)
    assert dones2.all()

    # a fresh reset clears the done flags
    venv.reset(seed=1)
    obs3, rewards3, dones3, infos3 = venv.step(actions)
    assert not dones3.all()


# ---------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------

def test_render_rgb_array():
    env = FlyAddictionEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
