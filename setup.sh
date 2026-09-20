#!/usr/bin/env bash
# Sets up flybrain-rl: clones the pinned upstream repos this project depends
# on, and creates the two virtualenvs used by the connectome sim / RL code
# (.venv) and the MuJoCo body renderer (.venv-body). Safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

clone_pinned() {
    local dir="$1" url="$2" sha="$3"
    if [ -d "$dir" ]; then
        echo "[setup] $dir already exists, skipping clone."
        return
    fi
    echo "[setup] Cloning $url @ $sha -> $dir"
    git clone "$url" "$dir"
    git -C "$dir" checkout "$sha"
}

echo "[setup] Cloning upstream repositories..."
clone_pinned browser-sim   https://github.com/snedea/flybrain.git       9191824d17871b7851645782d53d23f213ddb938
clone_pinned eon-fly-brain https://github.com/eonsystemspbc/fly-brain.git a3db62f9436074e485c0278290c2164ed6150808
clone_pinned flybody       https://github.com/TuragaLab/flybody.git     d015e9bfe441bd90ae431bac24c55cb74bdbce26
clone_pinned webgpu-fly    https://github.com/abgnydn/webgpu-fly.git    bb00419e874eee9e878542dcc5fce289ede2e1b9

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] ERROR: 'uv' is required (https://github.com/astral-sh/uv) but was not found on PATH." >&2
    exit 1
fi

if [ -d ".venv" ]; then
    echo "[setup] .venv already exists, skipping creation."
else
    echo "[setup] Creating .venv (python 3.11)..."
    uv venv --python 3.11 .venv
    uv pip install --python .venv/bin/python \
        torch numpy pandas pyarrow scipy tqdm joblib matplotlib gymnasium pytest
fi

if [ -d ".venv-body" ]; then
    echo "[setup] .venv-body already exists, skipping creation."
else
    echo "[setup] Creating .venv-body (python 3.11)..."
    uv venv --python 3.11 .venv-body
    uv pip install --python .venv-body/bin/python \
        -e ./flybody gymnasium "imageio[ffmpeg]" pillow
fi

cat <<'EOF'

[setup] Done.

Next steps:
  - FlyWire connectome annotations (~850 MB) are not fetched by this script.
    Download them with:
        bash webgpu-fly/tools/download_data.sh
  - The first time you construct a FastBrain (flyrl/fastbrain.py), it will
    build a ~300 MB synaptic-weight cache inside eon-fly-brain/data/. This
    only happens once; subsequent runs reuse the cache.
  - Activate an env with:
        source .venv/bin/activate        # connectome sim + RL
        source .venv-body/bin/activate   # MuJoCo body / 3D rendering
EOF
