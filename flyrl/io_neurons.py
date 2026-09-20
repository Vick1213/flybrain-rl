"""Anatomical input-group and readout-candidate index sets for BrainPolicy.

This module builds FIXED, DETERMINISTIC sets of local neuron indices (i.e.
indices into the (N,) neuron dimension used by FastBrain, obtained through
``fb.flyid_to_index``) from the FlyWire v783 neuron annotation table at
``webgpu-fly/data/raw/flywire_annotations/supplemental_files/
Supplemental_file1_neuron_annotations.tsv`` (read-only; see
``load_annotations``). It never modifies that file, nor anything under
eon-fly-brain/, browser-sim/, webgpu-fly/, flybody/ or fly3d/.

v1 of this module used the coarser ``browser-sim/data/classification.csv.gz``
table (only super_class/class/sub_class, no per-glomerulus or per-DN-type
identity), which is what produced the "same bilateral response whichever
side is stimulated" failure mode diagnosed in results/diag_readout.csv. v2
(this version) uses the richer annotation table's cell_type/cell_sub_class/
top_nt columns, and the specific anatomical populations it builds into each
input group were chosen empirically by scripts/screen_entry_points.py -- see
results/screen/io_v2_choice.json for the full justification.

Two kinds of index sets are built:

  1. Anatomical INPUT groups, one per FlyAddictionEnv observation channel
     (see ``GROUP_NAMES`` == ``FlyAddictionEnv.obs_channels``), each a
     specific empirically-chosen population (e.g. ALPN glomerulus DA2 for
     smoke_odor, visual_projection LC10e for reels_light -- see
     ``GROUP_MAX_RATE_HZ`` below and ``_build_groups_uncached``). Cached to
     ``flyrl/io_neurons.npz`` the first time they are built.

  2. READOUT candidates (all descending+motor neurons, with cell_type+side,
     plus DAN indices for logging) used by ``scripts/diag_readout.py`` to
     build the policy's 64-pool readout (cached to
     ``flyrl/readout_neurons.npz``).

All group construction is index-sorted (by FlyWire root_id) so it is fully
deterministic and reproducible without any RNG, EXCEPT the readout's generic
(non-named) pool assignment, which intentionally uses a seeded RNG (see
scripts/diag_readout.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = Path(__file__).resolve().parent / "io_neurons.npz"

# ---------------------------------------------------------------------
# v2 annotation source: the FlyWire v783 supplemental neuron-annotation
# table (see spec "Annotation source"). Far richer than the old
# browser-sim classification.csv.gz (root_id, flow, super_class,
# cell_class, cell_sub_class, cell_type, top_nt, side, ...) -- this is
# what lets us build per-glomerulus / per-DN-type / per-side entry points
# instead of v1's crude root_id-order left/right split. Read-only.
# ---------------------------------------------------------------------
_ANNOT_TSV = (_REPO_ROOT / "webgpu-fly" / "data" / "raw" / "flywire_annotations"
              / "supplemental_files" / "Supplemental_file1_neuron_annotations.tsv")

_annot_df_cache = None


def load_annotations():
    """Load (and cache in-process) the v783 neuron annotation table."""
    global _annot_df_cache
    if _annot_df_cache is None:
        _annot_df_cache = pd.read_csv(_ANNOT_TSV, sep="\t", low_memory=False)
    return _annot_df_cache


def local_indices_for_root_ids(fb, root_ids) -> np.ndarray:
    """root_ids (any iterable of int-like) -> sorted local indices, dropping
    any root_id absent from fb.flyid_to_index."""
    f2i = fb.flyid_to_index
    ids = sorted(int(i) for i in root_ids if int(i) in f2i)
    return np.array([f2i[i] for i in ids], dtype=np.int64)


def local_indices_for_mask(fb, df: pd.DataFrame, mask) -> np.ndarray:
    """Boolean mask over `df` (v2 annotation table) -> local indices."""
    return local_indices_for_root_ids(fb, df.loc[mask, "root_id"].tolist())

# Make sure eon-fly-brain/code (for benchmark.EXPERIMENTS) is importable,
# read-only, exactly like flyrl/fastbrain.py already does.
_EON_CODE = _REPO_ROOT / "eon-fly-brain" / "code"
if str(_EON_CODE) not in sys.path:
    sys.path.insert(0, str(_EON_CODE))

from benchmark import EXPERIMENTS  # noqa: E402  (read-only import)

SUGAR_GRN_IDS = tuple(EXPERIMENTS["sugar"]["neu_exc"])
# benchmark.py defines only 'sugar' and 'p9' experiments (no 'bitter'); the
# nicotine_taste group therefore falls back to the identifiable 'bitter'
# gustatory sub_class from the classification table (see module docstring
# of the spec: "bitter GRNs if identifiable else other gustatory neurons").

GROUP_NAMES = [
    "food_odor_L", "food_odor_R",
    "smoke_odor_L", "smoke_odor_R",
    "reels_light_L", "reels_light_R",
    "sugar_taste", "nicotine_taste", "reels_jackpot",
    "hunger", "nicotine", "withdrawal",
]

# ---------------------------------------------------------------------
# v2 anatomical entry-point choice (Task B), decided from the empirical
# screen in scripts/screen_entry_points.py -- see
# results/screen/io_v2_choice.json for the full data-backed justification,
# and results/screen/candidate_summary.csv / cosine_similarity.csv /
# mirror_consistency.csv / gating.csv / gating_v2_visual_ref.csv for the
# raw numbers. Headline reasoning (see io_v2_choice.json for detail):
#   - EVERY ORN glomerulus ignites the whole network (>40k spikes/300ms)
#     at rates as low as 1-2 Hz -- ORN input has no usable dynamic range.
#   - Most ALPN (projection-neuron) glomeruli are chaotically bistable
#     (ignition onset is not reproducible across RNG histories); only
#     ALPN glomerulus DA2 was confirmed non-igniting across 4 independent
#     trials spanning 1-150 Hz (weak signal, but the single reliable
#     olfactory channel found) -- used for smoke_odor. ALPN DC4 is the
#     best available SECOND olfactory channel (real DN response when
#     sub-ignition) but carries a documented, unresolved ignition risk --
#     used for food_odor with a conservative max rate.
#   - visual_projection NEVER ignites (69/69 types, both rates) and
#     several types are strongly, reproducibly lateralized (LC10e:
#     LI=0.85-0.97) -- used for reels_light (LC10e) and reels_jackpot
#     (MeTu1, pooled, anatomically distinct anterior-visual pathway).
#   - Endocrine neurosecretory cells (v1's hunger/nicotine/withdrawal)
#     give ZERO direct or gated DN effect -- replaced by octopaminergic
#     (hunger), DAN PAM (nicotine) and DAN PPL (withdrawal), all three of
#     which show real direct and/or gating effects on DN responsiveness.
GROUP_MAX_RATE_HZ = {
    "food_odor_L": 20.0, "food_odor_R": 20.0,      # ALPN DC4 -- conservative, ignition risk documented
    "smoke_odor_L": 150.0, "smoke_odor_R": 150.0,  # ALPN DA2 -- confirmed non-igniting 1-150 Hz
    "reels_light_L": 150.0, "reels_light_R": 150.0,  # visual_projection LC10e -- never ignites
    "sugar_taste": 200.0,     # unchanged from v1 / eon-fly-brain benchmark.py 'sugar' experiment
    "nicotine_taste": 150.0,  # pooled bitter GRNs
    "reels_jackpot": 150.0,   # visual_projection MeTu1, pooled
    "hunger": 150.0,          # octopaminergic, pooled
    "nicotine": 150.0,        # DAN PAM, pooled
    "withdrawal": 50.0,       # DAN PPL -- ignites at 150 Hz, capped
}


def _alpn_uniglomerular_ids(df: pd.DataFrame, glomerulus: str, side: str) -> list:
    alpn = df[(df["cell_class"] == "ALPN") & (df["cell_sub_class"] == "uniglomerular")
              & df["cell_type"].notna()]
    glom = alpn["cell_type"].str.split("_").str[0]
    return alpn.loc[(glom == glomerulus) & (alpn["side"] == side), "root_id"].tolist()


def _visual_type_ids(df: pd.DataFrame, cell_type: str, side: str = None) -> list:
    vis = df[df["super_class"] == "visual_projection"]
    mask = vis["cell_type"] == cell_type
    if side is not None:
        mask = mask & (vis["side"] == side)
    return vis.loc[mask, "root_id"].tolist()


def _build_groups_uncached(fb) -> dict:
    """v2 anatomical input groups -- see GROUP_MAX_RATE_HZ docstring above
    and results/screen/io_v2_choice.json for the full justification."""
    df = load_annotations()
    f2i = fb.flyid_to_index

    def idx(ids):
        return local_indices_for_root_ids(fb, ids)

    # food_odor: ALPN glomerulus DC4 (best available 2nd olfactory channel)
    food_L = idx(_alpn_uniglomerular_ids(df, "DC4", "left"))
    food_R = idx(_alpn_uniglomerular_ids(df, "DC4", "right"))

    # smoke_odor: ALPN glomerulus DA2 (the one robust, non-igniting olfactory channel)
    smoke_L = idx(_alpn_uniglomerular_ids(df, "DA2", "left"))
    smoke_R = idx(_alpn_uniglomerular_ids(df, "DA2", "right"))

    # reels_light: visual_projection LC10e (strongly lateralized, never ignites)
    reels_L = idx(_visual_type_ids(df, "LC10e", "left"))
    reels_R = idx(_visual_type_ids(df, "LC10e", "right"))

    # reels_jackpot: visual_projection MeTu1, pooled both sides (unsplit --
    # env's jackpot cue is a non-lateralized scalar flash); anatomically
    # distinct anterior-visual pathway from the LC10e looming/motion route.
    jackpot = idx(_visual_type_ids(df, "MeTu1", "left") + _visual_type_ids(df, "MeTu1", "right"))

    # sugar_taste: unchanged from v1 (matches eon-fly-brain benchmark.py 'sugar' experiment)
    sugar = idx(list(SUGAR_GRN_IDS))

    # nicotine_taste: pooled bitter GRNs (cell_sub_class == 'bitter'), both sides
    bitter_mask = df["cell_sub_class"] == "bitter"
    bitter = idx(df.loc[bitter_mask, "root_id"].tolist())

    # hunger: octopaminergic neurons (top_nt == 'octopamine'), pooled both sides
    octo_mask = (df["top_nt"] == "octopamine") & df["side"].isin(["left", "right"])
    hunger = idx(df.loc[octo_mask, "root_id"].tolist())

    # nicotine: DAN PAM cluster (reward/valence DANs), pooled both sides
    dan = df[df["cell_class"] == "DAN"].copy()
    dan["grp"] = dan["cell_type"].str.extract(r"^([A-Za-z]+)")
    nicotine = idx(dan.loc[dan["grp"] == "PAM", "root_id"].tolist())

    # withdrawal: DAN PPL cluster (punishment/aversive DANs), pooled both sides
    withdrawal = idx(dan.loc[dan["grp"] == "PPL", "root_id"].tolist())

    groups = {
        "food_odor_L": food_L, "food_odor_R": food_R,
        "smoke_odor_L": smoke_L, "smoke_odor_R": smoke_R,
        "reels_light_L": reels_L, "reels_light_R": reels_R,
        "sugar_taste": sugar, "nicotine_taste": bitter, "reels_jackpot": jackpot,
        "hunger": hunger, "nicotine": nicotine, "withdrawal": withdrawal,
    }
    return groups


def build_input_groups(fb, force_rebuild: bool = False) -> dict:
    """Return {group_name: np.ndarray[int64] of local neuron indices}.

    Cached to CACHE_PATH after first build (the underlying classification
    table and fb.flyid_to_index are both fixed, so this is deterministic).
    """
    if CACHE_PATH.exists() and not force_rebuild:
        data = np.load(CACHE_PATH)
        groups = {name: data[f"group_{name}"].astype(np.int64) for name in GROUP_NAMES}
        return groups

    groups = _build_groups_uncached(fb)

    # Sanity check: groups must be pairwise disjoint (encoder assumes each
    # input neuron belongs to exactly one anatomical group).
    seen = set()
    for name in GROUP_NAMES:
        idx = groups[name]
        overlap = seen.intersection(idx.tolist())
        assert not overlap, f"group {name} overlaps previously-built groups: {overlap}"
        seen.update(idx.tolist())

    to_save = {f"group_{name}": groups[name] for name in GROUP_NAMES}
    np.savez(CACHE_PATH, **to_save)
    return groups


def group_sizes(groups: dict) -> dict:
    return {name: int(len(groups[name])) for name in GROUP_NAMES}


def union_and_group_ids(groups: dict, group_names=GROUP_NAMES):
    """Flatten `groups` into (union_idx, group_id_per_neuron), both int64
    arrays of the same length, where group_id_per_neuron[k] is the index
    into `group_names` of the group that union_idx[k] belongs to."""
    all_idx, group_id = [], []
    for gi, name in enumerate(group_names):
        idx = groups[name]
        all_idx.append(idx)
        group_id.append(np.full(len(idx), gi, dtype=np.int64))
    union_idx = np.concatenate(all_idx) if all_idx else np.zeros(0, dtype=np.int64)
    group_id_per_neuron = np.concatenate(group_id) if group_id else np.zeros(0, dtype=np.int64)
    assert len(union_idx) == len(set(union_idx.tolist())), "overlap in input-group union"
    return union_idx, group_id_per_neuron


# ---------------------------------------------------------------------
# Readout candidates (descending+motor, DAN) -- used by diag_readout.py
# and scripts/screen_entry_points.py. Built from the v2 annotation table
# (see spec "Annotation source").
# ---------------------------------------------------------------------

def descending_motor_indices(fb):
    """Return (idx, side) for all descending+motor neurons present in the
    model; `side` is an array of 'left'/'right'/'center' strings aligned
    with `idx`, sorted by root_id ascending."""
    idx, side, _cell_type = descending_motor_with_type(fb)
    return idx, side


def descending_motor_with_type(fb):
    """Return (idx, side, cell_type) for all descending+motor neurons
    present in the model, sorted by root_id ascending. cell_type is the
    v2 annotation table's named DN/motor type (e.g. 'DNa01', 'DNp09',
    'MN10'), used for cell_type x side readout pooling and for reporting
    named-DN (DNa*/DNp*) responses in the screen."""
    df = load_annotations()
    mask = df["super_class"].isin(["descending", "motor"])
    sub = df.loc[mask, ["root_id", "side", "cell_type"]].copy()
    sub["root_id"] = sub["root_id"].astype(np.int64)
    f2i = fb.flyid_to_index
    sub = sub[sub["root_id"].isin(f2i.keys())].sort_values("root_id")
    idx = np.array([f2i[i] for i in sub["root_id"]], dtype=np.int64)
    side = sub["side"].astype(str).to_numpy()
    cell_type = sub["cell_type"].astype(str).to_numpy()
    return idx, side, cell_type


def dan_indices(fb):
    """DAN (dopaminergic) neuron local indices, sorted by root_id."""
    df = load_annotations()
    mask = df["cell_class"] == "DAN"
    return local_indices_for_mask(fb, df, mask)


def all_neuron_sides(fb):
    """Return an (N,) array of side strings ('left'/'right'/'center'/'nan'),
    aligned with local neuron index, for use when picking sides of any
    augmented readout neurons."""
    df = load_annotations()
    side = np.full(fb.N, "nan", dtype=object)
    f2i = fb.flyid_to_index
    root_ids = df["root_id"].astype(np.int64).to_numpy()
    sides = df["side"].astype(str).to_numpy()
    for rid, s in zip(root_ids, sides):
        li = f2i.get(int(rid))
        if li is not None:
            side[li] = s
    return side
