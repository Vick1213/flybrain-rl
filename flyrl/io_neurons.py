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

v3 (this version, Task 1) replaces BOTH odour channels with visual-
projection (VPN) channels too, so that ALL THREE directional sources
(food, smoke, reels) now enter the brain through the same anatomical
*class* of entry point. Rationale (see results/screen/io_v3_choice.json for
the full data-backed justification): a wider re-screen confirmed that
*every* ORN glomerulus ignites the whole network at rates as low as 1-2 Hz,
and ALPN (projection-neuron) ignition is a chaotic, RNG-history-dependent
threshold phenomenon that is not reproducible -- olfactory input is
degenerate/unusable for steering in this connectome (odour identity and
side are lost once the network ignites). visual_projection (VPN) types,
by contrast, never ignite (0/69 types tested, both 50 and 150 Hz) and
several are strongly and reproducibly lateralized. v3 therefore senses
food_odor and smoke_odor -- despite their env-level names, which are kept
unchanged as obs-channel identifiers only, see ``GROUP_NAMES`` -- through
two more VPN cell types (LPLC4 for food, LPC2 for smoke), chosen to
maximise |LI|, DN response magnitude, mirror consistency and pairwise
distinctness from each other, from reels_light's existing LC10e channel,
and from reels_jackpot's existing MeTu1 channel. Obs channel *names* are
unchanged (they are defined by flyrl.addiction_env.FlyAddictionEnv and
that file is not modified) -- only the anatomical entry-point neurons
change.

Two kinds of index sets are built:

  1. Anatomical INPUT groups, one per FlyAddictionEnv observation channel
     (see ``GROUP_NAMES`` == ``FlyAddictionEnv.obs_channels``), each a
     specific empirically-chosen population (e.g. visual_projection LPC2
     for smoke_odor, visual_projection LC10e for reels_light -- see
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
# v3 anatomical entry-point choice (Task 1), decided from the SAME
# empirical screen used for v2 (scripts/screen_entry_points.py) -- see
# results/screen/io_v3_choice.json for the full data-backed justification,
# and results/screen/candidate_summary.csv / cosine_similarity.csv /
# mirror_consistency.csv for the raw numbers. Headline reasoning:
#   - Olfactory input (ORN or ALPN, any glomerulus) is unusable for
#     steering in this connectome: ORN ignites the whole network at rates
#     as low as 1-2 Hz, and ALPN ignition onset is a chaotic/RNG-history-
#     dependent threshold phenomenon (see io_v2_choice.json). v3 drops
#     olfactory entry points entirely.
#   - visual_projection (VPN) NEVER ignites (0/69 types, 50 or 150 Hz) and
#     several types are strongly, reproducibly lateralized. v3 uses THREE
#     different VPN types, one per directional source: LC10e (reels_light,
#     unchanged from v2 -- the best-lateralized, best-established channel,
#     LI=0.85-0.97), LPLC4 (food_odor -- new in v3, the highest-magnitude
#     non-LC10-family VPN type with the lowest cross-talk to LC10e/MeTu1),
#     LPC2 (smoke_odor -- new in v3, the next-best-lateralized
#     non-LC10-family, non-MeTu1 VPN type). reels_jackpot keeps v2's MeTu1
#     (pooled, unsplit -- the jackpot cue is a non-lateralized scalar).
#     NOTE: every OTHER strongly-lateralized VPN type (LC10c-1/c-2/d/a/b)
#     is a member of the same LC10 "looming" family as LC10e and is highly
#     correlated with it (cosine similarity 0.6-0.95 same-side) -- these
#     were excluded from consideration as food/smoke channels precisely
#     because they would not be pairwise-distinct from reels_light. LPLC4
#     and LPC2 have lower |LI| than the LC10 family (0.47-0.79 and
#     0.62-0.68 respectively, vs LC10e's 0.85-0.97) but are the best
#     available trade-off against distinctness -- see io_v3_choice.json.
#   - Taste/jackpot/interoceptive channels are UNCHANGED from v2: sugar
#     GRNs (sugar_taste), pooled bitter GRNs (nicotine_taste), pooled
#     MeTu1 (reels_jackpot), octopaminergic (hunger), DAN PAM (nicotine),
#     DAN PPL @<=50Hz (withdrawal) -- see io_v2_choice.json for their
#     original justification, unaffected by the v3 odour->VPN change.
GROUP_MAX_RATE_HZ = {
    "food_odor_L": 150.0, "food_odor_R": 150.0,      # v3: visual_projection LPLC4 -- never ignites
    "smoke_odor_L": 150.0, "smoke_odor_R": 150.0,    # v3: visual_projection LPC2 -- never ignites
    "reels_light_L": 150.0, "reels_light_R": 150.0,  # visual_projection LC10e -- never ignites (unchanged)
    "sugar_taste": 200.0,     # unchanged from v1/v2 / eon-fly-brain benchmark.py 'sugar' experiment
    "nicotine_taste": 150.0,  # pooled bitter GRNs (unchanged from v2)
    "reels_jackpot": 150.0,   # visual_projection MeTu1, pooled (unchanged from v2)
    "hunger": 150.0,          # octopaminergic, pooled (unchanged from v2)
    "nicotine": 150.0,        # DAN PAM, pooled (unchanged from v2)
    "withdrawal": 50.0,       # DAN PPL -- ignites at 150 Hz, capped (unchanged from v2)
}


# Kept for reference/tests -- v2 used this to build the food_odor/smoke_odor
# ALPN groups (DC4/DA2); v3 no longer calls it (see _build_groups_uncached
# below), since io_v3_choice.json found olfactory input unusable.
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
    """v3 anatomical input groups -- see GROUP_MAX_RATE_HZ docstring above
    and results/screen/io_v3_choice.json for the full justification."""
    df = load_annotations()
    f2i = fb.flyid_to_index

    def idx(ids):
        return local_indices_for_root_ids(fb, ids)

    # food_odor: v3 -- visual_projection LPLC4 (odour is degenerate in this
    # model; LPLC4 is the highest-magnitude VPN type distinct from both
    # LC10e (reels_light) and MeTu1 (reels_jackpot); see io_v3_choice.json)
    food_L = idx(_visual_type_ids(df, "LPLC4", "left"))
    food_R = idx(_visual_type_ids(df, "LPLC4", "right"))

    # smoke_odor: v3 -- visual_projection LPC2 (next-best-lateralized VPN
    # type distinct from LC10e/MeTu1/LPLC4; see io_v3_choice.json)
    smoke_L = idx(_visual_type_ids(df, "LPC2", "left"))
    smoke_R = idx(_visual_type_ids(df, "LPC2", "right"))

    # reels_light: visual_projection LC10e (strongly lateralized, never ignites; unchanged from v2)
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
