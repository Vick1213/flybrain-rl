"""Scripted policies for FlyAddictionEnv, used to sanity-check the reward
tuning (see tests/test_addiction_env.py) and to print a small behavioural
table via ``python -m flyrl.scripted``.

All policies steer using the same bilateral (left/right antenna) klinotaxis
trick: turn toward whichever sensor reads stronger, and slow down as the
target's intensity approaches its max (i.e. as the fly gets close), which
keeps the fly hovering near a source once it arrives instead of overshooting
in and out of the consumption radius every step.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from flyrl.addiction_env import FlyAddictionEnv, AddictionParams, addiction_metrics

CHANNEL_INDEX = {name: i for i, name in enumerate(FlyAddictionEnv.obs_channels)}

_SOURCE_CHANNELS = {
    "food": ("food_odor_L", "food_odor_R"),
    "smoke": ("smoke_odor_L", "smoke_odor_R"),
    "reels": ("reels_light_L", "reels_light_R"),
}


def _steer_towards(obs: np.ndarray, info: Optional[dict], left_key: str, right_key: str,
                    target_name: str, turn_gain: float = 60.0,
                    lost_signal_eps: float = 1e-3) -> np.ndarray:
    """Bilateral klinotaxis: turn toward whichever sensor reads stronger.

    Two refinements keep this robust against the environment's realistic
    sensor noise (odor falloff over distance, reels flicker, a visual field
    of view):
      - If both sensors read ~0 (target out of range / out of the visual
        field of view), the intensity gradient carries no directional
        information, so the fly instead rotates in place ("search") until
        it picks the target back up, rather than driving blindly forward.
      - Once ``info`` confirms the fly is already consuming ``target_name``,
        it hovers (near-zero forward speed) instead of throttling speed
        from the (flicker-noisy) intensity reading, which otherwise can
        misjudge proximity and overshoot out of the consumption radius.
    """
    L = float(obs[CHANNEL_INDEX[left_key]])
    R = float(obs[CHANNEL_INDEX[right_key]])

    if max(L, R) < lost_signal_eps:
        # No directional signal at all: spin in place to reacquire target.
        return np.array([1.0, -1.0], dtype=np.float32)

    turn = float(np.clip(turn_gain * (L - R), -1.0, 1.0))

    if info is not None and info.get("at") == target_name:
        speed_action = -1.0  # hover: already consuming the target
    else:
        proximity = max(L, R)
        speed_frac = float(np.clip(1.0 - proximity, 0.2, 1.0))
        speed_action = 2.0 * speed_frac - 1.0
    return np.array([turn, speed_action], dtype=np.float32)


def _idle_action() -> np.ndarray:
    return np.array([0.0, -1.0], dtype=np.float32)


class ForagerPolicy:
    """Steers to FOOD only while hunger exceeds a threshold; idles (no
    turning, no forward motion) otherwise."""

    def __init__(self, hunger_threshold: float = 0.3):
        self.hunger_threshold = hunger_threshold

    def reset(self):
        pass

    def act(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        hunger = float(obs[CHANNEL_INDEX["hunger"]])
        if hunger > self.hunger_threshold:
            return _steer_towards(obs, info, *_SOURCE_CHANNELS["food"], target_name="food")
        return _idle_action()


class SmokerPolicy:
    """Always steers toward and stays at SMOKE."""

    def reset(self):
        pass

    def act(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        return _steer_towards(obs, info, *_SOURCE_CHANNELS["smoke"], target_name="smoke")


class ReelsPolicy:
    """Always steers toward and stays at REELS."""

    def reset(self):
        pass

    def act(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        return _steer_towards(obs, info, *_SOURCE_CHANNELS["reels"], target_name="reels")


class GreedyPolicy:
    """Myopic epsilon-greedy bandit over the 3 sources: maintains an
    exponential-moving-average estimate of the reward obtained while at each
    source, and steers toward the current best estimate (random source with
    probability epsilon). This is meant to be lured into smoking early
    (high initial hedonic reward at zero tolerance) before its own reward
    feedback teaches it otherwise.
    """

    def __init__(self, epsilon: float = 0.1, ema_alpha: float = 0.2,
                 seed: Optional[int] = None):
        self.epsilon = epsilon
        self.ema_alpha = ema_alpha
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        # Slight optimism so every source gets tried early on.
        self.value = {"food": 0.1, "smoke": 0.1, "reels": 0.1}
        self.target = "food"

    def update(self, at: Optional[str], reward: float):
        """Call after env.step() with info['at'] and the step reward."""
        if at in self.value:
            self.value[at] = (1.0 - self.ema_alpha) * self.value[at] + self.ema_alpha * reward

    def act(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        if self.rng.uniform() < self.epsilon:
            self.target = self.rng.choice(list(self.value.keys()))
        else:
            self.target = max(self.value, key=self.value.get)
        return _steer_towards(obs, info, *_SOURCE_CHANNELS[self.target], target_name=self.target)


def run_episode(env: FlyAddictionEnv, policy, seed: Optional[int] = None):
    """Run one episode of `policy` on `env`, returning (total_reward, infos)."""
    obs, info = env.reset(seed=seed)
    policy.reset()
    infos = []
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        if hasattr(policy, "update"):
            policy.update(info.get("at"), reward)
        infos.append(info)
        total_reward += reward
    return total_reward, infos


def _mean_reward_thirds(infos: list) -> tuple:
    """Return (mean reward in first third, mean reward in last third)."""
    n = len(infos)
    third = max(1, n // 3)
    rewards = np.array([info["reward"] for info in infos], dtype=np.float64)
    first = float(np.mean(rewards[:third]))
    last = float(np.mean(rewards[-third:]))
    return first, last


def main(n_seeds: int = 20, params: Optional[AddictionParams] = None):
    policy_factories = {
        "Forager": lambda: ForagerPolicy(),
        "Smoker": lambda: SmokerPolicy(),
        "Reels": lambda: ReelsPolicy(),
        "Greedy": lambda: GreedyPolicy(seed=0),
    }

    rows = []
    for name, factory in policy_factories.items():
        returns = []
        metrics_list = []
        first_thirds, last_thirds = [], []
        for seed in range(n_seeds):
            env = FlyAddictionEnv(params=params)
            policy = factory()
            total_reward, infos = run_episode(env, policy, seed=seed)
            returns.append(total_reward)
            metrics_list.append(addiction_metrics(infos))
            f, l = _mean_reward_thirds(infos)
            first_thirds.append(f)
            last_thirds.append(l)

        mean_return = float(np.mean(returns))
        agg = {}
        for key in ("frac_food", "frac_smoke", "frac_reels", "compulsion_smoke",
                    "compulsion_reels", "mean_withdrawal", "final_tolerance"):
            agg[key] = float(np.mean([m[key] for m in metrics_list]))
        rows.append({
            "policy": name,
            "mean_return": mean_return,
            "mean_reward_first_third": float(np.mean(first_thirds)),
            "mean_reward_last_third": float(np.mean(last_thirds)),
            **agg,
        })

    cols = ["policy", "mean_return", "mean_reward_first_third", "mean_reward_last_third",
            "frac_food", "frac_smoke", "frac_reels", "compulsion_smoke",
            "compulsion_reels", "mean_withdrawal", "final_tolerance"]
    widths = {c: max(len(c), 10) for c in cols}
    header = " ".join(f"{c:>{widths[c]}}" for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = []
        for c in cols:
            v = row[c]
            if isinstance(v, str):
                line.append(f"{v:>{widths[c]}}")
            else:
                line.append(f"{v:>{widths[c]}.3f}")
        print(" ".join(line))
    return rows


if __name__ == "__main__":
    main()
