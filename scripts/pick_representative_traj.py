"""Picks the "most representative" episode among a run's saved trajectories.

Used by scripts/render_trained.sh after `flyrl.evaluate` has written
results/<run>/traj_seed*.npz for several episodes: prints (to stdout) the
path of the one whose `--metric` fraction (default frac_smoke, i.e. the
fraction of steps with at == 'smoke') is closest to the mean of that
fraction across all of that run's saved trajectories -- a simple "closest
to the run's own average behaviour" pick, not the best- or worst-case
episode.

Usage
-----
    .venv/bin/python scripts/pick_representative_traj.py results/addicted
    .venv/bin/python scripts/pick_representative_traj.py results/sober --metric frac_food
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

AT_CODE = {"food": 1, "smoke": 2, "reels": 3}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="results/<run> directory containing traj_seed*.npz")
    parser.add_argument("--metric", default="frac_smoke",
                        choices=[f"frac_{k}" for k in AT_CODE])
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.run_dir, "traj_seed*.npz")))
    if not paths:
        raise SystemExit(f"pick_representative_traj: no traj_seed*.npz found in {args.run_dir!r} "
                          f"-- run flyrl.evaluate first")

    code = AT_CODE[args.metric.split("_", 1)[1]]
    fracs = {}
    for path in paths:
        data = np.load(path)
        fracs[path] = float(np.mean(data["at"] == code))

    mean_frac = float(np.mean(list(fracs.values())))
    best = min(fracs, key=lambda p: abs(fracs[p] - mean_frac))
    print(best)


if __name__ == "__main__":
    main()
