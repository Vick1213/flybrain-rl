"""flyrl.evaluate: evaluate a trained BrainPolicy checkpoint on FlyAddictionEnv.

Usage
-----
    python -m flyrl.evaluate --ckpt results/addicted/ckpt.npz --episodes 8

Prints a metrics table in the same format as flyrl.scripted's main() output
(return, mean reward in the first/last third of the episode, fraction of
time at each source, compulsion indices, final tolerance), plus the mean DAN
firing rate while the fly is at smoke / at reels / at food / elsewhere.

Trajectory file format
-----------------------
Also saves one file per evaluated episode to
``results/<run_name>/traj_seed<k>.npz`` (``<run_name>`` = the checkpoint's
parent directory name, ``<k>`` = that episode's env seed), meant to be
consumed directly by a 3D renderer. Each file has these arrays, one entry
per simulated env.step() (length T = the episode length used at train time,
``config["n_steps"]``, read from the checkpoint):

    x, y            float32 (T,)  fly position each step (unit-square arena)
    heading         float32 (T,)  fly heading theta, radians
    at              int32   (T,)  0 = none, 1 = food, 2 = smoke, 3 = reels
    h, n, tau, w    float32 (T,)  hunger, nicotine, tolerance, withdrawal
    jackpot         bool    (T,)  True on reels-jackpot steps
    reward          float32 (T,)  per-step total reward
    dan_rate_hz     float32 (T,)  mean DAN population firing rate (Hz)
    source_food, source_smoke, source_reels   float32 (2,) each: the fixed
        xy position of that source for this episode (constant over T)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flyrl.addiction_env import VecFlyAddictionEnv, addiction_metrics
from flyrl.policy import BrainPolicy

AT_CODE = {None: 0, "food": 1, "smoke": 2, "reels": 3}

METRIC_KEYS = ("frac_food", "frac_smoke", "frac_reels", "compulsion_smoke",
               "compulsion_reels", "mean_withdrawal", "final_tolerance")


def _load_ckpt(ckpt_path: Path):
    data = np.load(ckpt_path)
    config = json.loads(str(data["config_json"]))
    mean_theta = data["mean_theta"].astype(np.float32)
    return config, mean_theta


def run_eval(ckpt_path, episodes: int, seed_base: int = 2000, save_traj: bool = True):
    ckpt_path = Path(ckpt_path)
    run_dir = ckpt_path.parent
    config, mean_theta = _load_ckpt(ckpt_path)

    policy = BrainPolicy(batch=episodes, device=config.get("device", "cpu"),
                          dt=config.get("dt", 0.5), steps_per_action=config.get("steps_per_action", 20),
                          seed=config.get("seed", 0) + 777)
    n_steps = int(config.get("n_steps", 300))
    env = VecFlyAddictionEnv(num_envs=episodes, n_steps=n_steps)

    theta = np.tile(mean_theta, (episodes, 1))
    policy.set_params(theta)
    policy.reset()
    obs = env.reset(seed=seed_base)

    B, T = episodes, n_steps
    traj = {
        "x": np.zeros((B, T), dtype=np.float32),
        "y": np.zeros((B, T), dtype=np.float32),
        "heading": np.zeros((B, T), dtype=np.float32),
        "at": np.zeros((B, T), dtype=np.int32),
        "h": np.zeros((B, T), dtype=np.float32),
        "n": np.zeros((B, T), dtype=np.float32),
        "tau": np.zeros((B, T), dtype=np.float32),
        "w": np.zeros((B, T), dtype=np.float32),
        "jackpot": np.zeros((B, T), dtype=bool),
        "reward": np.zeros((B, T), dtype=np.float32),
        "dan_rate_hz": np.zeros((B, T), dtype=np.float32),           # non-input ("other") DANs only
        "dan_rate_driven_hz": np.zeros((B, T), dtype=np.float32),    # directly-driven (PAM+PPL input) DANs only
    }
    dan_by_at = {None: [], "food": [], "smoke": [], "reels": []}
    episode_infos = [[] for _ in range(B)]

    for t in range(T):
        actions = policy.act(obs)
        obs, rewards, dones, infos = env.step(actions)
        dan_rate = policy.mean_dan_rate_hz()
        dan_rate_driven = policy.dan_rate_driven_hz()
        for i, sub_env in enumerate(env.envs):
            traj["x"][i, t] = sub_env.pos[0]
            traj["y"][i, t] = sub_env.pos[1]
            traj["heading"][i, t] = sub_env.theta
        for i, info in enumerate(infos):
            at = info.get("at") if info else None
            traj["at"][i, t] = AT_CODE.get(at, 0)
            traj["h"][i, t] = info.get("h", 0.0) if info else 0.0
            traj["n"][i, t] = info.get("n", 0.0) if info else 0.0
            traj["tau"][i, t] = info.get("tau", 0.0) if info else 0.0
            traj["w"][i, t] = info.get("w", 0.0) if info else 0.0
            traj["jackpot"][i, t] = bool(info.get("jackpot", False)) if info else False
            traj["reward"][i, t] = float(rewards[i])
            traj["dan_rate_hz"][i, t] = float(dan_rate[i])
            traj["dan_rate_driven_hz"][i, t] = float(dan_rate_driven[i])
            if info:
                episode_infos[i].append(info)
                dan_by_at[at].append(float(dan_rate[i]))

    metrics_list = [addiction_metrics(episode_infos[i]) for i in range(B)]
    returns = [float(np.sum(traj["reward"][i])) for i in range(B)]
    third = max(1, T // 3)
    first_thirds = [float(np.mean(traj["reward"][i, :third])) for i in range(B)]
    last_thirds = [float(np.mean(traj["reward"][i, -third:])) for i in range(B)]

    if save_traj:
        for i, sub_env in enumerate(env.envs):
            seed_k = seed_base + i
            out = {k: v[i] for k, v in traj.items()}
            out["source_food"] = sub_env.sources["food"].astype(np.float32)
            out["source_smoke"] = sub_env.sources["smoke"].astype(np.float32)
            out["source_reels"] = sub_env.sources["reels"].astype(np.float32)
            np.savez(run_dir / f"traj_seed{seed_k}.npz", **out)

    return metrics_list, returns, first_thirds, last_thirds, dan_by_at


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=2000)
    parser.add_argument("--no-traj", action="store_true")
    args = parser.parse_args(argv)

    metrics_list, returns, first_thirds, last_thirds, dan_by_at = run_eval(
        args.ckpt, args.episodes, seed_base=args.seed_base, save_traj=not args.no_traj)

    run_name = Path(args.ckpt).parent.name
    row = {
        "policy": run_name,
        "mean_return": float(np.mean(returns)),
        "mean_reward_first_third": float(np.mean(first_thirds)),
        "mean_reward_last_third": float(np.mean(last_thirds)),
    }
    for key in METRIC_KEYS:
        row[key] = float(np.mean([m[key] for m in metrics_list]))
    for at_key, label in ((None, "dan_elsewhere"), ("food", "dan_food"),
                           ("smoke", "dan_smoke"), ("reels", "dan_reels")):
        vals = dan_by_at.get(at_key, [])
        row[label] = float(np.mean(vals)) if vals else float("nan")

    cols = ["policy", "mean_return", "mean_reward_first_third", "mean_reward_last_third",
            "frac_food", "frac_smoke", "frac_reels", "compulsion_smoke", "compulsion_reels",
            "mean_withdrawal", "final_tolerance", "dan_food", "dan_smoke", "dan_reels", "dan_elsewhere"]
    widths = {c: max(len(c), 10) for c in cols}
    header = " ".join(f"{c:>{widths[c]}}" for c in cols)
    print(header)
    print("-" * len(header))
    line = []
    for c in cols:
        v = row[c]
        line.append(f"{v:>{widths[c]}}" if isinstance(v, str) else f"{v:>{widths[c]}.3f}")
    print(" ".join(line))
    return row


if __name__ == "__main__":
    main()
