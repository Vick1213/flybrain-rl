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
used the richer annotation table's cell_type/cell_sub_class/top_nt columns
for interoceptive/taste/jackpot channels, but still routed food_odor/
smoke_odor through olfactory ALPN glomeruli (DC4/DA2) -- see
results/screen/io_v2_choice.json.

v3 replaced BOTH odour channels with visual-projection (VPN) channels too,
so that all three directional sources (food, smoke, reels) entered the
brain through the same anatomical *class* of entry point, but each source
still had its OWN dedicated lateralized population (LPLC4 for food, LPC2
for smoke, LC10e for reels) -- see results/screen/io_v3_choice.json. With
I/O v3, the food channel (LPLC4, the largest-magnitude of the three, ~115
descending/motor neurons driven per side) steered through the brain as
well as the scripted teacher, but the smoke channel (LPC2, only ~17 DNs
driven) and reels channel (LC10e) steered much worse -- an ability
imbalance that would confound the addiction experiment (a fly that can
only walk to food looks "sober" for the wrong reason, not because it
values food over drugs).

v4 (this version, Task 1) fixes that imbalance structurally: ALL THREE
directional sources now drive the SAME shared lateralized population pair
-- visual_projection LPLC4, full left/right populations (the single
highest-magnitude, never-igniting VPN channel found in the v2/v3 screens,
see io_v3_choice.json) -- instead of each source getting its own
anatomical population. The LPC2 (v3 smoke) and LC10e (v3 reels) groups are
DROPPED entirely; the trainable encoder (flyrl.policy) is responsible for
combining all three sources' L/R intensities (plus internal state) into a
single drive_L/drive_R pair that sets this one shared population's firing
rate, so "which source I steer toward" becomes a question of encoder
*valuation* weights (trainable, and hence where addiction/preference can
live) rather than which anatomical channel happens to carry the strongest
signal. Contact/interoceptive channels (sugar_taste, nicotine_taste,
reels_jackpot, hunger, nicotine, withdrawal) are UNCHANGED from v2/v3.
Obs channel *names* are unchanged (they are defined by
flyrl.addiction_env.FlyAddictionEnv and that file is not modified) --
FlyAddictionEnv still emits food_odor_L/R, smoke_odor_L/R, reels_light_L/R
etc. as before; only the anatomical entry-point neurons and how many
distinct anatomical groups obs channels map onto have changed.

Two kinds of index sets are built:

  1. Anatomical INPUT groups (see ``GROUP_NAMES`` below) -- v4 has 8, not
     one per obs channel: the two lateralized steering groups ``steer_L``/
     ``steer_R`` (visual_projection LPLC4, shared by all three directional
     sources -- the encoder decides how each source's L/R obs intensities
     and the internal state channels combine into this one pair's drive)
     plus the six unchanged own-channel contact/interoceptive groups
     (sugar_taste, nicotine_taste, reels_jackpot, hunger, nicotine,
     withdrawal). Cached to ``flyrl/io_neurons.npz`` the first time they
     are built.

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
    "steer_L", "steer_R",
    "sugar_taste", "nicotine_taste", "reels_jackpot",
    "hunger", "nicotine", "withdrawal",
]

# ---------------------------------------------------------------------
# v4 anatomical entry-point choice (Task 1). steer_L/steer_R replace v3's
# THREE separate lateralized pairs (food_odor LPLC4, smoke_odor LPC2,
# reels_light LC10e) with ONE shared pair: visual_projection LPLC4, full
# left/right populations -- the highest-magnitude, never-igniting VPN
# channel found in the v2/v3 screens (see results/screen/io_v3_choice.json:
# 123/107 responsive DN/motor neurons at 150Hz, vs LPC2's 18/16 and
# LC10e's 45/56 -- LPLC4 was already the strongest of the three v3
# channels). Rationale for pooling onto one channel instead of keeping
# three (see results/screen/io_v4_choice.json): v3 measured that the
# food channel (LPLC4) steered through the brain as well as the scripted
# teacher (100% reach) while smoke (LPC2, ~17 DNs/side) and reels (LC10e)
# reached only 47-62% and 16-22% respectively -- an anatomical-capacity
# imbalance, not a preference difference, that would confound the
# addiction experiment. Routing all three sources through the SAME
# high-capacity population and letting the trainable encoder (flyrl.policy)
# combine their L/R intensities + internal state into one drive_L/drive_R
# pair equalizes steering *ability* across sources, so any behavioural bias
# ES later learns reflects trained *valuation* weights, not which
# anatomical channel happened to reach more descending neurons.
#   - LPC2 (v3 smoke_odor) and LC10e (v3 reels_light) are DROPPED --
#     removed as anatomical entry points entirely (still confirmed distinct
#     from LPLC4 and MeTu1 by the v3 screen's cosine-similarity numbers,
#     but simply unused in v4).
#   - Taste/jackpot/interoceptive channels are UNCHANGED from v2/v3: sugar
#     GRNs (sugar_taste), pooled bitter GRNs (nicotine_taste), pooled
#     MeTu1 (reels_jackpot), octopaminergic (hunger), DAN PAM (nicotine),
#     DAN PPL @<=50Hz (withdrawal).
GROUP_MAX_RATE_HZ = {
    "steer_L": 150.0, "steer_R": 150.0,      # v4: visual_projection LPLC4 (shared), never ignites
    "sugar_taste": 200.0,     # unchanged from v1/v2/v3 / eon-fly-brain benchmark.py 'sugar' experiment
    "nicotine_taste": 150.0,  # pooled bitter GRNs (unchanged)
    "reels_jackpot": 150.0,   # visual_projection MeTu1, pooled (unchanged)
    "hunger": 150.0,          # octopaminergic, pooled (unchanged)
    "nicotine": 150.0,        # DAN PAM, pooled (unchanged)
    "withdrawal": 50.0,       # DAN PPL -- ignites at 150 Hz, capped (unchanged)
}


# Kept for reference/tests -- v2 used this to build the food_odor/smoke_odor
# ALPN groups (DC4/DA2); v3/v4 no longer call it, since io_v3_choice.json
# found olfactory input unusable.
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
    """v4 anatomical input groups -- see GROUP_MAX_RATE_HZ docstring above
    and results/screen/io_v4_choice.json for the full justification."""
    df = load_annotations()

    def idx(ids):
        return local_indices_for_root_ids(fb, ids)

    # steer_L/steer_R: v4 -- visual_projection LPLC4, full populations,
    # shared by all three directional sources (food/smoke/reels); the
    # trainable encoder (flyrl.policy) combines each source's L/R obs
    # intensity + internal state into this one pair's drive.
    steer_L = idx(_visual_type_ids(df, "LPLC4", "left"))
    steer_R = idx(_visual_type_ids(df, "LPLC4", "right"))

    # reels_jackpot: visual_projection MeTu1, pooled both sides (unsplit --
    # env's jackpot cue is a non-lateralized scalar flash); anatomically
    # distinct anterior-visual pathway from the LPLC4 route (unchanged).
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
        "steer_L": steer_L, "steer_R": steer_R,
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
        try:
            data = np.load(CACHE_PATH)
            groups = {name: data[f"group_{name}"].astype(np.int64) for name in GROUP_NAMES}
            return groups
        except KeyError:
            pass  # stale cache from an earlier io_neurons version -- rebuild below

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
