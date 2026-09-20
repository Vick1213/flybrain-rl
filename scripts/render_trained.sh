#!/usr/bin/env bash
# Evaluates one or more trained BrainPolicy checkpoints and renders each
# one's most representative episode as a 3D mp4 (a BRAIN-DRIVEN fly, via
# fly3d.scene.render_trajectory), plus -- when exactly two runs are given
# -- a side-by-side comparison video.
#
# Usage
# -----
#   scripts/render_trained.sh [run_name ...]      # default: addicted sober
#
# For each <run_name>, expects results/<run_name>/ckpt.npz (a
# flyrl.train_es checkpoint) and does, in order:
#   1. `flyrl.evaluate` (.venv, $EPISODES episodes, --threads $THREADS) --
#      writes results/<run_name>/traj_seed*.npz
#   2. picks the most representative episode -- the one whose
#      frac_smoke is closest to that run's own mean across the evaluated
#      episodes (scripts/pick_representative_traj.py)
#   3. renders it to renders/<run_name>_trained.mp4 via
#      `fly3d.scene --mode traj` (.venv-body, MUJOCO_GL=glfw), captioned
#      per the TITLE case below
#
# If exactly two runs were (successfully) rendered, also builds a
# side-by-side comparison video renders/<run1>_vs_<run2>.mp4: the first
# run's own most-representative episode, and the SAME seed's trajectory
# for the second run when results/<run2>/traj_seed<seed>.npz exists
# (falling back to the second run's own most-representative episode
# otherwise) -- two 640x720 panels hstacked (scripts/make_comparison_video.py),
# shorter clip padded to match.
#
# Env var overrides: THREADS (torch threads for evaluate, default 4 --
# kept low on purpose so this doesn't starve a concurrent flyrl.train_es
# run), EPISODES (default 8), FPS (default 30), FRAMES_PER_STEP (default 3).
#
# Example (smoke-testing against a scratch checkpoint copy, see the task
# report for why the real results/addicted/ckpt.npz is copied aside
# first rather than read directly while training is writing it):
#   scripts/render_trained.sh _preview_addicted
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ "$#" -gt 0 ]; then
    RUNS=("$@")
else
    RUNS=(addicted sober)
fi

THREADS="${THREADS:-4}"
EPISODES="${EPISODES:-8}"
FPS="${FPS:-30}"
FRAMES_PER_STEP="${FRAMES_PER_STEP:-3}"

# bash 3.2 (macOS system bash) has neither associative arrays nor
# ${var^^}, hence the case statement instead of a TITLES map.
title_for() {
    case "$1" in
        addicted) printf '%s' "ADDICTED FLY — trained on hijacked dopamine signal" ;;
        sober)    printf '%s' "SOBER FLY — trained on true welfare" ;;
        *)        printf '%s FLY' "$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')" ;;
    esac
}

mkdir -p renders
TMPDIR_CMP="$(mktemp -d "${TMPDIR:-/tmp}/render_trained.XXXXXX")"
trap 'rm -rf "$TMPDIR_CMP"' EXIT

# Parallel arrays (indexed by successfully-rendered run, in order) --
# not associative arrays, for bash 3.2 compatibility.
RENDERED_RUNS=()
RENDERED_OUT=()
RENDERED_TRAJ=()

for RUN in "${RUNS[@]}"; do
    CKPT="results/${RUN}/ckpt.npz"
    if [ ! -f "$CKPT" ]; then
        echo "skipping ${RUN}: ${CKPT} not found" >&2
        continue
    fi

    echo "== ${RUN}: evaluate (${EPISODES} episodes, ${THREADS} threads) =="
    .venv/bin/python -m flyrl.evaluate --ckpt "$CKPT" --episodes "$EPISODES" --threads "$THREADS"

    TRAJ="$(.venv/bin/python scripts/pick_representative_traj.py "results/${RUN}")"
    echo "== ${RUN}: most representative episode -> ${TRAJ} =="

    TITLE="$(title_for "$RUN")"
    OUT="renders/${RUN}_trained.mp4"
    echo "== ${RUN}: render ${OUT} =="
    MUJOCO_GL=glfw .venv-body/bin/python -m fly3d.scene --mode traj \
        --traj "$TRAJ" --out "$OUT" --title "$TITLE" \
        --fps "$FPS" --frames-per-step "$FRAMES_PER_STEP"

    RENDERED_RUNS+=("$RUN")
    RENDERED_OUT+=("$OUT")
    RENDERED_TRAJ+=("$TRAJ")
done

if [ "${#RENDERED_RUNS[@]}" -eq 2 ]; then
    A="${RENDERED_RUNS[0]}";    B="${RENDERED_RUNS[1]}"
    A_OUT="${RENDERED_OUT[0]}"; B_OUT="${RENDERED_OUT[1]}"
    A_TRAJ="${RENDERED_TRAJ[0]}"

    SEED_A="$(basename "$A_TRAJ" | sed -E 's/traj_seed([0-9]+)\.npz/\1/')"
    MATCHED_B="results/${B}/traj_seed${SEED_A}.npz"

    B_CMP="$B_OUT"
    if [ -f "$MATCHED_B" ]; then
        echo "== comparison: reusing seed ${SEED_A} (${A}'s representative episode) for ${B} too =="
        B_CMP="${TMPDIR_CMP}/${B}_seed${SEED_A}.mp4"
        MUJOCO_GL=glfw .venv-body/bin/python -m fly3d.scene --mode traj \
            --traj "$MATCHED_B" --out "$B_CMP" --title "$(title_for "$B")" \
            --fps "$FPS" --frames-per-step "$FRAMES_PER_STEP"
    else
        echo "== comparison: seed ${SEED_A} not available for ${B}; using its own most-representative episode instead ==" >&2
    fi

    COMPARISON="renders/${A}_vs_${B}.mp4"
    echo "== side-by-side: ${COMPARISON} =="
    .venv-body/bin/python scripts/make_comparison_video.py "$A_OUT" "$B_CMP" "$COMPARISON"
else
    echo "comparison video skipped: need exactly 2 successfully-rendered runs, got ${#RENDERED_RUNS[@]}" >&2
fi

echo "done."
