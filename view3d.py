"""Interactive MuJoCo viewer for the fly-addiction 3D scene (fly3d/scene.py).

The fly is animated kinematically (qpos written directly + mj_forward every
frame; there is no physics stepping), driven either by the same
hand-authored staged trajectory used for renders/staged_demo.mp4, or by a
scripted policy running the real 2D FlyAddictionEnv.

On macOS, mujoco.viewer.launch_passive needs to run on the process's main
thread inside Apple's window-server-aware runloop, which is what the
`mjpython` launcher (installed alongside the `mujoco` package) provides.
Plain `python` will raise/crash on macOS for this. Launch with:

    .venv-body/bin/mjpython view3d.py                              # staged demo loop (repeats)
    .venv-body/bin/mjpython view3d.py --policy greedy --seed 0     # scripted episode
    .venv-body/bin/mjpython view3d.py --max-seconds 5              # smoke-test: exits itself

Mouse: drag to orbit, scroll to zoom, ctrl+drag to pan -- MuJoCo's own
passive-viewer camera, independent of our scripted 'orbit'/'closeup'/'top'
cameras used for offscreen rendering in scene.py.

NOTE: this file only exercises `mujoco.viewer.launch_passive` for a few
seconds via `--max-seconds`, which is checked automatically. The real
interactive mouse-orbit experience (an open GUI window a person drags with
a mouse) cannot be verified by this script or its author -- see the task
report.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from fly3d.scene import (
    AddictionScene3D,
    _STAGE_FOOD_XY, _STAGE_SMOKE_XY, _STAGE_REELS_XY,
    _staged_segments, _state_at, _wrap_angle,
    load_flyrl_modules, _make_policy,
)


def run_staged_loop(scene: AddictionScene3D, max_seconds: float | None):
    """Loop the ~12s staged trajectory (walk -> smoke -> walk -> scroll)
    forever, in real time, until the viewer window is closed (or
    `max_seconds` elapses, for automated smoke-testing).
    """
    scene.set_sources(_STAGE_FOOD_XY, _STAGE_SMOKE_XY, _STAGE_REELS_XY)
    segs = _staged_segments(scene)
    total_t = segs[-1]["t1"]
    dt = 1.0 / 60.0

    start = time.time()
    t = 0.0
    with mujoco.viewer.launch_passive(scene.m, scene.d) as viewer:
        while viewer.is_running():
            loop_t0 = time.time()
            st = _state_at(segs, t % total_t)
            scene.set_state(**st)
            if scene._reels_names:
                viewer.update_texture(scene.ids.reels_tex_id)
            viewer.sync()

            if max_seconds is not None and (time.time() - start) >= max_seconds:
                break
            t += dt
            time.sleep(max(0.0, dt - (time.time() - loop_t0)))


def run_episode_loop(scene: AddictionScene3D, policy_name: str, seed: int,
                     max_seconds: float | None, fps: int = 30, frames_per_step: int = 3):
    """Drive the viewer from one scripted-policy FlyAddictionEnv episode,
    in real time, interpolating `frames_per_step` frames per env step."""
    addiction_env, _ = load_flyrl_modules()
    env = addiction_env.FlyAddictionEnv()
    policy = _make_policy(policy_name, seed)

    obs, info = env.reset(seed=seed)
    policy.reset()
    scene.set_sources(env.sources["food"], env.sources["smoke"], env.sources["reels"])

    prev_x, prev_y, prev_h = float(env.pos[0]), float(env.pos[1]), float(env.theta)
    t = 0.0
    dt_frame = 1.0 / fps
    start = time.time()
    terminated = truncated = False

    with mujoco.viewer.launch_passive(scene.m, scene.d) as viewer:
        while viewer.is_running() and not (terminated or truncated):
            action = policy.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            if hasattr(policy, "update"):
                policy.update(info.get("at"), reward)

            new_x, new_y, new_h = float(env.pos[0]), float(env.pos[1]), float(env.theta)
            dh = _wrap_angle(new_h - prev_h)
            for k in range(frames_per_step):
                loop_t0 = time.time()
                frac = (k + 1) / frames_per_step
                ix = prev_x + (new_x - prev_x) * frac
                iy = prev_y + (new_y - prev_y) * frac
                ih = _wrap_angle(prev_h + dh * frac)
                t += dt_frame
                is_last = k == frames_per_step - 1
                scene.set_state(ix, iy, ih, at=info.get("at"),
                                hunger=info.get("h", 0.0), nicotine=info.get("n", 0.0),
                                tolerance=info.get("tau", 0.0),
                                jackpot=bool(info.get("jackpot", False)) and is_last, t=t)
                if scene._reels_names:
                    viewer.update_texture(scene.ids.reels_tex_id)
                viewer.sync()

                if max_seconds is not None and (time.time() - start) >= max_seconds:
                    return
                if not viewer.is_running():
                    return
                time.sleep(max(0.0, dt_frame - (time.time() - loop_t0)))
            prev_x, prev_y, prev_h = new_x, new_y, new_h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=None,
                        choices=["forager", "smoker", "reels", "greedy"],
                        help="scripted policy to run on the real 2D env; "
                             "omit to loop the hand-authored staged demo instead")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="exit automatically after this many seconds "
                             "(used for headless smoke-testing; interactive "
                             "use should omit this)")
    args = parser.parse_args()

    scene = AddictionScene3D(create_renderer=False)
    if args.policy is None:
        run_staged_loop(scene, max_seconds=args.max_seconds)
    else:
        run_episode_loop(scene, args.policy, args.seed, max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
