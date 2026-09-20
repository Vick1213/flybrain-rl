"""FlyAddictionEnv: a small, pure-numpy Gymnasium environment modelling a fruit
fly that can become "addicted" to smoking (nicotine vapour) and to scrolling
reels (a flickering screen with a variable-ratio jackpot schedule), competing
with ordinary feeding.

This is a playful computational-neuroscience toy. It is meant to eventually be
driven by a frozen fly-connectome spiking network acting as the policy, with
this env's observation channels designed to map onto real sensory neuron
populations (bilateral antennae/eyes + interoceptive channels), so that we can
measure whether the learned policy becomes behaviourally "compulsive".

Everything here is pure numpy except the optional matplotlib-based
`rgb_array` renderer.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

SOURCE_NAMES = ("food", "smoke", "reels")


def _wrap_angle(theta: float) -> float:
    """Wrap an angle (radians) to (-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class AddictionParams:
    """All tunable constants for :class:`FlyAddictionEnv`, grouped by
    subsystem. Distances are fractions of the unit-square arena side, angles
    are radians unless a field name says otherwise, and one "step" is one
    ``env.step()`` call (nominally ~20ms x N of simulated connectome brain
    time once a real policy drives it; the env itself is unit-less).
    """

    # ---------------- episode / world ----------------
    n_steps: int = 300
    """Episode length in steps before truncation."""

    arena_margin: float = 0.12
    """Minimum distance from any arena wall used when placing sources and
    the fly's initial position, so nothing spawns flush against an edge."""

    min_source_separation: float = 0.32
    """Minimum pairwise Euclidean distance enforced between the 3 sources
    (food/smoke/reels) when they are placed at reset."""

    r_consume: float = 0.06
    """Radius around a source's center within which the fly is considered
    to be consuming it (eating / smoking / watching reels)."""

    # ---------------- motion ----------------
    v_max: float = 0.045
    """Maximum forward speed, in arena-widths per step, at action[1] == 1."""

    turn_rate_max: float = 0.5
    """Maximum turn, in radians per step, at |action[0]| == 1."""

    # ---------------- sensors ----------------
    sensor_offset: float = 0.035
    """Perpendicular distance from the fly's body center to each of the two
    (left/right) antenna/eye sensor points. Bilateral comparison of the two
    sensors' readings is what allows klinotaxis-style steering."""

    odor_lambda: float = 0.25
    """Exponential length-scale of odor/light intensity falloff with
    distance: intensity = exp(-distance / odor_lambda)."""

    reels_fov_deg: float = 120.0
    """Half-angle (degrees), from a sensor's forward-facing direction,
    within which the (visual) reels source is visible to that sensor;
    outside this angle the reels light reads as zero on that side."""

    reels_flicker_range: tuple = (0.5, 1.0)
    """Uniform range that the reels visual intensity (both sides, same
    factor) is multiplied by every step to model screen flicker."""

    # ---------------- hunger / food ----------------
    hunger_init: float = 0.5
    """Hunger h at episode start."""

    hunger_rise_rate: float = 0.006
    """Per-step increase in hunger h (applied every step, before eating)."""

    hunger_eat_rate: float = 0.02
    """Per-step decrease in hunger while eating at FOOD."""

    starvation_penalty: float = 0.05
    """Extra reward penalty applied every step that hunger is saturated at
    its maximum (h == 1)."""

    food_reward_scale: float = 1.2
    """Reward while eating = food_reward_scale * h, i.e. eating while sated
    (h near 0) is worth ~nothing, eating while hungry is worth a lot."""

    # ---------------- nicotine / tolerance / withdrawal ----------------
    nicotine_gain_rate: float = 0.15
    """Per-step nicotine increase while at SMOKE, saturating toward 1:
    n <- n + nicotine_gain_rate * (1 - n)."""

    nicotine_decay_rate: float = 0.06
    """Exponential per-step decay rate of nicotine away from SMOKE:
    n <- n * exp(-nicotine_decay_rate)."""

    tolerance_rise_rate: float = 0.02
    """Per-step tolerance increase, proportional to the current nicotine
    level, so tolerance accumulates with cumulative nicotine exposure."""

    tolerance_decay_rate: float = 0.0015
    """Slow constant per-step decay of tolerance (recovery over time)."""

    withdrawal_k: float = 0.4
    """Scale relating nicotine to relief of tolerance-driven withdrawal:
    w = max(0, tolerance - withdrawal_k * nicotine). Kept below 1 so that
    once tolerance exceeds withdrawal_k, even fully-saturated nicotine
    (n == 1, i.e. continuously smoking) cannot fully relieve withdrawal --
    a heavy smoker who never stops still feels some withdrawal, since ever
    more nicotine would be needed to keep up with rising tolerance."""

    smoke_hedonic_gain: float = 1.3
    """Peak hedonic reward gain while smoking at zero tolerance; realized
    hedonic reward is smoke_hedonic_gain * (1 - tolerance) (classic
    opponent-process: shrinks as tolerance grows)."""

    withdrawal_penalty_scale: float = 0.8
    """Coefficient c_w on the per-step withdrawal penalty -c_w * w, applied
    every step regardless of the fly's location."""

    # ---------------- reels / variable-ratio schedule ----------------
    reels_jackpot_prob: float = 0.15
    """Base probability p of a jackpot payout on any step spent at REELS."""

    reels_jackpot_reward: float = 1.0
    """Reward paid out on a jackpot step, before the habituation discount."""

    reels_habituation_rise: float = 0.05
    """Per-step increase in reels habituation while at REELS."""

    reels_habituation_decay: float = 0.02
    """Per-step decrease in reels habituation while away from REELS."""

    reels_habituation_effect: float = 0.7
    """Fraction by which jackpot probability and jackpot magnitude are both
    discounted at full habituation (habituation == 1)."""

    # ---------------- movement cost ----------------
    energy_cost_scale: float = 0.01
    """Coefficient on the per-step movement energy penalty, normalized by
    v_max: r_energy = -energy_cost_scale * (forward_speed / v_max)."""

    # ---------------- rendering ----------------
    render_size: int = 240
    """Side length, in pixels, of the square rgb_array render."""


class FlyAddictionEnv(gym.Env):
    """A continuous 2D arena with three fixed sources (FOOD, SMOKE, REELS).

    Action: Box(2,) in [-1, 1] = (turn_rate, forward_speed_command); forward
    speed command is linearly mapped to [0, v_max].

    Observation: Box(12,), float32, all channels scaled to [0, 1]. See
    `FlyAddictionEnv.obs_channels` for the channel order.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    obs_channels = [
        "food_odor_L", "food_odor_R",
        "smoke_odor_L", "smoke_odor_R",
        "reels_light_L", "reels_light_R",
        "sugar_taste", "nicotine_taste", "reels_jackpot",
        "hunger", "nicotine", "withdrawal",
    ]

    def __init__(self, params: Optional[AddictionParams] = None,
                 render_mode: Optional[str] = None, **param_overrides: Any):
        super().__init__()
        if params is None:
            params = AddictionParams()
        if param_overrides:
            params = dataclasses.replace(params, **param_overrides)
        self.params = params

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)

        # runtime state, populated by reset()
        self.pos = np.zeros(2, dtype=np.float64)
        self.theta = 0.0
        self.sources: dict[str, np.ndarray] = {}
        self.h = float(self.params.hunger_init)
        self.n = 0.0
        self.tau = 0.0
        self.reels_habituation = 0.0
        self.t = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        p = self.params
        rng = self.np_random

        low, high = p.arena_margin, 1.0 - p.arena_margin
        pts = rng.uniform(low, high, size=(3, 2))
        for _ in range(1000):
            d01 = np.linalg.norm(pts[0] - pts[1])
            d02 = np.linalg.norm(pts[0] - pts[2])
            d12 = np.linalg.norm(pts[1] - pts[2])
            if min(d01, d02, d12) >= p.min_source_separation:
                break
            pts = rng.uniform(low, high, size=(3, 2))
        self.sources = {
            "food": pts[0].copy(),
            "smoke": pts[1].copy(),
            "reels": pts[2].copy(),
        }

        self.pos = rng.uniform(0.15, 0.85, size=2).astype(np.float64)
        self.theta = float(rng.uniform(-np.pi, np.pi))

        self.h = float(p.hunger_init)
        self.n = 0.0
        self.tau = 0.0
        self.reels_habituation = 0.0
        self.t = 0

        obs = self._get_obs(at=None, jackpot=False)
        info = self._get_info(
            reward_terms={
                "r_food": 0.0, "r_smoke_hedonic": 0.0, "r_withdrawal": 0.0,
                "r_reels": 0.0, "r_energy": 0.0, "r_starvation": 0.0, "reward": 0.0,
            },
            at=None, jackpot=False,
        )
        return obs, info

    def step(self, action):
        p = self.params
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        turn_a, speed_a = float(action[0]), float(action[1])

        turn = turn_a * p.turn_rate_max
        speed = (speed_a + 1.0) * 0.5 * p.v_max

        self.theta = _wrap_angle(self.theta + turn)
        self.pos = self.pos + speed * np.array([np.cos(self.theta), np.sin(self.theta)])
        self.pos = np.clip(self.pos, 0.0, 1.0)

        # which source (if any) is the fly currently consuming
        dists = {name: float(np.linalg.norm(self.pos - src)) for name, src in self.sources.items()}
        at = None
        nearest = min(dists, key=dists.get)
        if dists[nearest] <= p.r_consume:
            at = nearest

        # ---- hunger / food ----
        self.h = float(np.clip(self.h + p.hunger_rise_rate, 0.0, 1.0))
        r_food = 0.0
        if at == "food":
            r_food = p.food_reward_scale * self.h
            self.h = float(np.clip(self.h - p.hunger_eat_rate, 0.0, 1.0))
        r_starvation = -p.starvation_penalty if self.h >= 1.0 - 1e-9 else 0.0

        # ---- nicotine / tolerance / withdrawal ----
        if at == "smoke":
            self.n = float(np.clip(self.n + p.nicotine_gain_rate * (1.0 - self.n), 0.0, 1.0))
        else:
            self.n = float(self.n * np.exp(-p.nicotine_decay_rate))
        self.tau = float(np.clip(
            self.tau + p.tolerance_rise_rate * self.n - p.tolerance_decay_rate, 0.0, 1.0))
        w = max(0.0, self.tau - p.withdrawal_k * self.n)
        r_smoke_hedonic = p.smoke_hedonic_gain * (1.0 - self.tau) if at == "smoke" else 0.0
        r_withdrawal = -p.withdrawal_penalty_scale * w

        # ---- reels / variable-ratio schedule ----
        if at == "reels":
            self.reels_habituation = float(np.clip(
                self.reels_habituation + p.reels_habituation_rise, 0.0, 1.0))
        else:
            self.reels_habituation = float(np.clip(
                self.reels_habituation - p.reels_habituation_decay, 0.0, 1.0))
        jackpot = False
        r_reels = 0.0
        if at == "reels":
            discount = 1.0 - p.reels_habituation_effect * self.reels_habituation
            eff_p = p.reels_jackpot_prob * discount
            if self.np_random.uniform() < eff_p:
                jackpot = True
                r_reels = p.reels_jackpot_reward * discount

        # ---- movement energy cost ----
        r_energy = -p.energy_cost_scale * (speed / max(p.v_max, 1e-9))

        reward = r_food + r_smoke_hedonic + r_withdrawal + r_reels + r_energy + r_starvation

        self.t += 1
        terminated = False
        truncated = self.t >= p.n_steps

        obs = self._get_obs(at=at, jackpot=jackpot)
        reward_terms = {
            "r_food": r_food,
            "r_smoke_hedonic": r_smoke_hedonic,
            "r_withdrawal": r_withdrawal,
            "r_reels": r_reels,
            "r_energy": r_energy,
            "r_starvation": r_starvation,
            "reward": reward,
        }
        info = self._get_info(reward_terms=reward_terms, at=at, jackpot=jackpot)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return self._render_rgb_array()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _sensor_positions(self):
        p = self.params
        left_dir = self.theta + np.pi / 2.0
        right_dir = self.theta - np.pi / 2.0
        left = self.pos + p.sensor_offset * np.array([np.cos(left_dir), np.sin(left_dir)])
        right = self.pos + p.sensor_offset * np.array([np.cos(right_dir), np.sin(right_dir)])
        return left, right

    def _bilateral_intensity(self, source_pos: np.ndarray, fov_half_rad: Optional[float] = None):
        p = self.params
        left, right = self._sensor_positions()
        out = []
        for sensor_pos in (left, right):
            d = source_pos - sensor_pos
            dist = float(np.linalg.norm(d))
            intensity = float(np.exp(-dist / p.odor_lambda))
            if fov_half_rad is not None:
                bearing = _wrap_angle(float(np.arctan2(d[1], d[0])) - self.theta)
                if abs(bearing) > fov_half_rad:
                    intensity = 0.0
            out.append(intensity)
        return out[0], out[1]

    def _get_obs(self, at: Optional[str], jackpot: bool) -> np.ndarray:
        p = self.params
        food_L, food_R = self._bilateral_intensity(self.sources["food"])
        smoke_L, smoke_R = self._bilateral_intensity(self.sources["smoke"])
        fov_half = np.radians(p.reels_fov_deg)
        reels_L, reels_R = self._bilateral_intensity(self.sources["reels"], fov_half_rad=fov_half)

        flicker = float(self.np_random.uniform(p.reels_flicker_range[0], p.reels_flicker_range[1]))
        reels_L *= flicker
        reels_R *= flicker

        sugar_taste = 1.0 if at == "food" else 0.0
        nicotine_taste = 1.0 if at == "smoke" else 0.0
        reels_jackpot_ch = 1.0 if jackpot else 0.0

        w = max(0.0, self.tau - p.withdrawal_k * self.n)

        obs = np.array([
            food_L, food_R, smoke_L, smoke_R, reels_L, reels_R,
            sugar_taste, nicotine_taste, reels_jackpot_ch,
            self.h, self.n, np.clip(w, 0.0, 1.0),
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _get_info(self, reward_terms: dict, at: Optional[str], jackpot: bool) -> dict:
        info = dict(reward_terms)
        w = max(0.0, self.tau - self.params.withdrawal_k * self.n)
        info.update({
            "at": at,
            "h": self.h,
            "n": self.n,
            "tau": self.tau,
            "w": w,
            "jackpot": jackpot,
            "t": self.t,
        })
        return info

    def _render_rgb_array(self) -> np.ndarray:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        p = self.params
        size_in = p.render_size / 100.0
        fig = plt.figure(figsize=(size_in, size_in), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, color="#eef2f5", zorder=0))

        colors = {"food": "#2ecc71", "smoke": "#7f8c8d", "reels": "#8e44ad"}
        for name, src in self.sources.items():
            ax.add_patch(patches.Circle(src, p.r_consume, color=colors[name], alpha=0.35, zorder=1))
            ax.plot(src[0], src[1], marker="o", color=colors[name], markersize=7, zorder=2)

        # fly as a small triangle pointing along heading
        L = 0.035
        tip = self.pos + L * np.array([np.cos(self.theta), np.sin(self.theta)])
        back_l = self.pos + 0.6 * L * np.array([
            np.cos(self.theta + 2.5), np.sin(self.theta + 2.5)])
        back_r = self.pos + 0.6 * L * np.array([
            np.cos(self.theta - 2.5), np.sin(self.theta - 2.5)])
        ax.add_patch(patches.Polygon([tip, back_l, back_r], closed=True, color="black", zorder=3))

        # interoceptive state bars (hunger, nicotine, tolerance) along the top
        bars = [("h", self.h, "#e67e22"), ("n", self.n, "#c0392b"), ("tau", self.tau, "#2980b9")]
        bar_w, bar_h, gap = 0.28, 0.035, 0.02
        for i, (_label, val, color) in enumerate(bars):
            x0 = 0.02 + i * (bar_w + gap)
            y0 = 1.0 - bar_h - 0.02
            ax.add_patch(patches.Rectangle((x0, y0), bar_w, bar_h, fill=False,
                                            edgecolor="#333333", linewidth=0.8, zorder=4))
            ax.add_patch(patches.Rectangle((x0, y0), bar_w * float(np.clip(val, 0, 1)), bar_h,
                                            color=color, zorder=5))

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img = buf[:, :, :3].copy()
        plt.close(fig)
        return img


class VecFlyAddictionEnv:
    """Minimal batched wrapper holding B independent FlyAddictionEnv
    instances (a plain Python loop under the hood -- no vectorized physics).

    ``reset(seed)`` returns obs of shape (B, 12). ``step(actions)`` with
    ``actions`` of shape (B, 2) returns (obs, rewards, dones, infos) with
    obs (B, 12), rewards (B,), dones (B,) and infos a length-B list of dicts.

    There is no auto-reset: once a sub-env finishes (terminated or
    truncated) it stays "done" -- it keeps returning its last observation,
    zero reward, and done=True -- until :meth:`reset` is called again for
    the whole batch.
    """

    def __init__(self, num_envs: int, params: Optional[AddictionParams] = None,
                 **param_overrides: Any):
        self.num_envs = int(num_envs)
        self.envs = [FlyAddictionEnv(params=params, **param_overrides)
                     for _ in range(self.num_envs)]
        self._done = np.zeros(self.num_envs, dtype=bool)
        self._last_obs = [np.zeros(12, dtype=np.float32) for _ in range(self.num_envs)]

    def reset(self, seed: Optional[int] = None):
        obs_list = []
        for i, env in enumerate(self.envs):
            sub_seed = None if seed is None else int(seed) + i
            obs, _info = env.reset(seed=sub_seed)
            obs_list.append(obs)
        self._last_obs = obs_list
        self._done[:] = False
        return np.stack(obs_list, axis=0)

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float64)
        obs_list, rewards, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            if self._done[i]:
                obs_list.append(self._last_obs[i])
                rewards.append(0.0)
                dones.append(True)
                infos.append({})
                continue
            obs, reward, terminated, truncated, info = env.step(actions[i])
            done = bool(terminated or truncated)
            self._done[i] = done
            self._last_obs[i] = obs
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return (
            np.stack(obs_list, axis=0),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )


def addiction_metrics(infos: list) -> dict:
    """Summarize a single episode's per-step info dicts (as produced by
    ``FlyAddictionEnv.step``) into addiction-relevant behavioural metrics.

    Returns a dict with:
      - frac_food / frac_smoke / frac_reels / frac_none: fraction of steps
        spent consuming each source (or none).
      - compulsion_smoke / compulsion_reels: fraction of the *high-hunger*
        steps (h > 0.7) spent at smoke/reels respectively -- consuming the
        drug/reels despite being very hungry.
      - mean_withdrawal: mean of the withdrawal signal w over the episode.
      - final_tolerance: tolerance tau at the last step.
      - time_to_first_smoke: index (0-based) of the first step at which the
        fly was at SMOKE, or None if it never smoked.
    """
    n = len(infos)
    if n == 0:
        return {
            "frac_food": 0.0, "frac_smoke": 0.0, "frac_reels": 0.0, "frac_none": 1.0,
            "compulsion_smoke": 0.0, "compulsion_reels": 0.0,
            "mean_withdrawal": 0.0, "final_tolerance": 0.0, "time_to_first_smoke": None,
        }

    at_arr = [info.get("at") for info in infos]
    h_arr = np.array([info.get("h", 0.0) for info in infos], dtype=np.float64)
    w_arr = np.array([info.get("w", 0.0) for info in infos], dtype=np.float64)

    def frac(name):
        return sum(1 for a in at_arr if a == name) / n

    high_hunger = h_arr > 0.7
    denom = int(high_hunger.sum())

    def compulsion(name):
        if denom == 0:
            return 0.0
        mask = np.array([a == name for a in at_arr])
        return float(np.sum(mask & high_hunger) / denom)

    time_to_first_smoke = None
    for i, a in enumerate(at_arr):
        if a == "smoke":
            time_to_first_smoke = i
            break

    return {
        "frac_food": frac("food"),
        "frac_smoke": frac("smoke"),
        "frac_reels": frac("reels"),
        "frac_none": frac(None),
        "compulsion_smoke": compulsion("smoke"),
        "compulsion_reels": compulsion("reels"),
        "mean_withdrawal": float(np.mean(w_arr)),
        "final_tolerance": float(infos[-1].get("tau", 0.0)),
        "time_to_first_smoke": time_to_first_smoke,
    }
