"""Task A: empirical screen of candidate anatomical sensory/neuromodulatory
entry points into the frozen whole-fly-brain connectome (see spec "Task A --
empirical screen").

For every candidate population (an ORN glomerulus type, a uniglomerular ALPN
glomerulus, a Johnston's-organ subtype, a visual_projection cell type, a
gustatory cell type, or a neuromodulatory/interoceptive population), split by
side where the anatomy defines a side, we drive that population ALONE at
50 Hz and 150 Hz for 300 ms (dt=0.5 ms, 600 steps) and record how the
descending+motor (DN/motor) population and the DAN population respond,
split ipsi/contra relative to the stimulated side.

Candidates are batched 32-per-FastBrain-call (`fb.run(rates (B,K), ...)`
lets each of the 32 batch ELEMENTS receive an independent stimulus, since
FastBrain shares connectome weights but not activity across the batch dim).

Outputs (all under results/screen/):
  candidate_summary.csv   -- one row per (candidate, side, rate)
  cosine_similarity.csv   -- pairwise cosine similarity of DN response
                             vectors (150 Hz) between every screened
                             candidate x side
  mirror_consistency.csv  -- per (family, subtype) correlation between the
                             left-stim DN response and the side-swapped
                             right-stim DN response (matched by cell_type+side)
  gating.csv              -- interoceptive candidates that evoked ZERO DN
                             spikes alone, co-stimulated with a reference
                             food-like channel, vs food-alone

This script only *reads* the v2 annotation table (via flyrl.io_neurons) and
FastBrain; it writes only under results/screen/.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flyrl.fastbrain import FastBrain  # noqa: E402
from flyrl.io_neurons import (  # noqa: E402
    load_annotations, local_indices_for_root_ids, descending_motor_with_type,
    dan_indices, SUGAR_GRN_IDS,
)

OUT_DIR = _REPO_ROOT / "results" / "screen"

DT_MS = 0.5
WINDOW_MS = 300.0
N_STEPS = int(round(WINDOW_MS / DT_MS))  # 600
RATES = (50.0, 150.0)
IGNITION_THRESHOLD = 40_000
WAVE_SIZE = 32

# Spec-suggested candidate glomerulus sets for the pooled ALPN food-like /
# smoke-like populations (and to steer the reference "food channel" used
# for the interoceptive gating test toward a biologically sensible choice).
FOOD_GLOMERULI = {"DM1", "DM2", "DM4", "VA2"}
SMOKE_GLOMERULI = {"DL5", "V", "DA2"}

GATE_FOOD_RATE = 100.0
GATE_STIM_RATE = 150.0


def _side_tag(side: str) -> str:
    return side[0].upper()


# ---------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------

def build_candidates(fb) -> list[dict]:
    df = load_annotations()
    f2i = fb.flyid_to_index
    cands = []

    def add(name, family, subtype, side, root_ids, min_n=1):
        root_ids = [int(r) for r in root_ids]
        idx = local_indices_for_root_ids(fb, root_ids)
        if len(idx) < min_n:
            return
        cands.append(dict(
            name=name, family=family, subtype=subtype, side=side,
            local_idx=idx, n_neurons=int(len(idx)), n_annotated=len(root_ids),
        ))

    # --- ORN glomerulus types, >=20/side ---
    olf = df[(df["cell_class"] == "olfactory") & df["cell_type"].notna()
             & df["cell_type"].str.startswith("ORN_")]
    for ct, grp in olf.groupby("cell_type"):
        for side in ("left", "right"):
            ids = grp.loc[grp["side"] == side, "root_id"].tolist()
            if len(ids) >= 20:
                add(f"ORN_{ct[4:]}_{_side_tag(side)}", "ORN", ct, side, ids)

    # --- uniglomerular ALPN, grouped by glomerulus, >=3/side ---
    alpn = df[(df["cell_class"] == "ALPN") & (df["cell_sub_class"] == "uniglomerular")
              & df["cell_type"].notna()].copy()
    alpn["glom"] = alpn["cell_type"].str.split("_").str[0]
    for glom, grp in alpn.groupby("glom"):
        for side in ("left", "right"):
            ids = grp.loc[grp["side"] == side, "root_id"].tolist()
            if len(ids) >= 3:
                add(f"ALPN_{glom}_{_side_tag(side)}", "ALPN", glom, side, ids)

    # --- pooled ALPN food-like / smoke-like ---
    for label, gloms in (("ALPN_food_pool", FOOD_GLOMERULI), ("ALPN_smoke_pool", SMOKE_GLOMERULI)):
        sub = alpn[alpn["glom"].isin(gloms)]
        for side in ("left", "right"):
            ids = sub.loc[sub["side"] == side, "root_id"].tolist()
            add(f"{label}_{_side_tag(side)}", "ALPN_pool", label, side, ids, min_n=1)

    # --- Johnston's organ subtypes, >=5/side ---
    jo = df[df["cell_type"].notna() & df["cell_type"].str.startswith("JO-")
            & (df["cell_type"] != "JO-unclear")]
    for ct, grp in jo.groupby("cell_type"):
        for side in ("left", "right"):
            ids = grp.loc[grp["side"] == side, "root_id"].tolist()
            if len(ids) >= 5:
                add(f"{ct}_{_side_tag(side)}", "JO", ct, side, ids)

    # --- visual_projection types, >=40/side ---
    vis = df[df["super_class"] == "visual_projection"]
    for ct, grp in vis.groupby("cell_type"):
        for side in ("left", "right"):
            ids = grp.loc[grp["side"] == side, "root_id"].tolist()
            if len(ids) >= 40:
                add(f"VIS_{ct}_{_side_tag(side)}", "visual", ct, side, ids)

    # --- gustatory: every cell_type, both sides (no size floor -- report n) ---
    gust = df[df["cell_class"] == "gustatory"]
    for ct, grp in gust.groupby("cell_type"):
        for side in ("left", "right"):
            ids = grp.loc[grp["side"] == side, "root_id"].tolist()
            if ids:
                add(f"GUST_{ct}_{_side_tag(side)}", "gustatory", ct, side, ids)
    # dedicated sugar-GRN benchmark set (unsplit; matches flyrl.io_neurons.SUGAR_GRN_IDS)
    add("sugar_GRN_benchmark", "gustatory_sugar", "sugar", None, list(SUGAR_GRN_IDS))
    # pooled bitter GRNs (cell_sub_class == 'bitter'), by side
    bitter = df[df["cell_sub_class"] == "bitter"]
    for side in ("left", "right"):
        ids = bitter.loc[bitter["side"] == side, "root_id"].tolist()
        add(f"bitter_pool_{_side_tag(side)}", "gustatory_bitter", "bitter", side, ids)

    # --- endocrine (interoceptive) cell types, by side ---
    endo = df[df["super_class"] == "endocrine"]
    for ct, grp in endo.groupby("cell_type"):
        for side in ("left", "right"):
            ids = grp.loc[grp["side"] == side, "root_id"].tolist()
            if ids:
                add(f"ENDO_{ct}_{_side_tag(side)}", "endocrine", ct, side, ids)

    # --- octopaminergic (top_nt == octopamine), by side ---
    octo = df[df["top_nt"] == "octopamine"]
    for side in ("left", "right"):
        ids = octo.loc[octo["side"] == side, "root_id"].tolist()
        add(f"octopaminergic_{_side_tag(side)}", "neuromod", "octopamine", side, ids)

    # --- central serotonergic (top_nt == serotonin & super_class == central) ---
    sero = df[(df["top_nt"] == "serotonin") & (df["super_class"] == "central")]
    for side in ("left", "right"):
        ids = sero.loc[sero["side"] == side, "root_id"].tolist()
        add(f"serotonergic_central_{_side_tag(side)}", "neuromod", "serotonin", side, ids)

    # --- DAN subtypes PAM vs PPL*, by side ---
    dan = df[df["cell_class"] == "DAN"].copy()
    dan["grp"] = dan["cell_type"].str.extract(r"^([A-Za-z]+)")
    for grp_name in ("PAM", "PPL"):
        sub = dan[dan["grp"] == grp_name]
        for side in ("left", "right"):
            ids = sub.loc[sub["side"] == side, "root_id"].tolist()
            add(f"DAN_{grp_name}_{_side_tag(side)}", "neuromod", f"DAN_{grp_name}", side, ids)

    return cands


# ---------------------------------------------------------------------
# Simulation waves
# ---------------------------------------------------------------------

def run_wave(fb, wave: list[tuple[dict, float]]):
    """wave: list of (candidate_dict, rate), len(wave) <= fb.batch. Returns
    (len(wave), fb.N) counts (a short/partial wave is zero-padded up to
    fb.batch rows before simulating, since FastBrain's batch size is fixed
    at construction, then trimmed back down)."""
    assert len(wave) <= fb.batch
    union_idx = np.unique(np.concatenate([c["local_idx"] for c, _r in wave]))
    col_pos = {int(v): i for i, v in enumerate(union_idx.tolist())}
    K = len(union_idx)
    rates_mat = np.zeros((fb.batch, K), dtype=np.float32)
    for row, (c, r) in enumerate(wave):
        cols = [col_pos[int(v)] for v in c["local_idx"].tolist()]
        rates_mat[row, cols] = r

    fb.reset()
    fb.set_input_neurons(union_idx)
    fb.set_exc_indices(union_idx)
    rates_t = torch.as_tensor(rates_mat)
    counts = fb.run(rates_t, N_STEPS)
    return counts.numpy()[:len(wave)]


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ---------------------------------------------------------------------
# Response summarization
# ---------------------------------------------------------------------

def summarize_row(cand, rate, row_counts, dm_idx, dm_side, dm_ct, dan_idx):
    total_spikes_all = float(row_counts.sum())
    dm_counts = row_counts[dm_idx]
    dan_counts = row_counts[dan_idx]

    left_m = dm_side == "left"
    right_m = dm_side == "right"
    center_m = ~(left_m | right_m)

    dna_m = np.char.startswith(dm_ct.astype(str), "DNa")
    dnp_m = np.char.startswith(dm_ct.astype(str), "DNp")
    dna01_m = dm_ct == "DNa01"
    dna02_m = dm_ct == "DNa02"
    dnp09_m = dm_ct == "DNp09"

    def side_spikes(mask, side_mask):
        m = mask & side_mask
        return float(dm_counts[m].sum())

    row = {
        "name": cand["name"], "family": cand["family"], "subtype": cand["subtype"],
        "stim_side": cand["side"] if cand["side"] else "NA",
        "rate_hz": rate,
        "n_neurons_stim": cand["n_neurons"], "n_annotated": cand["n_annotated"],
        "total_spikes_all": total_spikes_all,
        "ignition": bool(total_spikes_all > IGNITION_THRESHOLD),
        "dm_n_responsive": int((dm_counts > 0).sum()),
        "dm_spikes_total": float(dm_counts.sum()),
        "dm_spikes_left": float(dm_counts[left_m].sum()),
        "dm_spikes_right": float(dm_counts[right_m].sum()),
        "dm_spikes_center": float(dm_counts[center_m].sum()),
        "dm_n_resp_left": int((dm_counts[left_m] > 0).sum()),
        "dm_n_resp_right": int((dm_counts[right_m] > 0).sum()),
        "dm_n_resp_center": int((dm_counts[center_m] > 0).sum()),
        "DNa01_left": side_spikes(dna01_m, left_m), "DNa01_right": side_spikes(dna01_m, right_m),
        "DNa02_left": side_spikes(dna02_m, left_m), "DNa02_right": side_spikes(dna02_m, right_m),
        "DNa_other_left": side_spikes(dna_m & ~dna01_m & ~dna02_m, left_m),
        "DNa_other_right": side_spikes(dna_m & ~dna01_m & ~dna02_m, right_m),
        "DNp09_left": side_spikes(dnp09_m, left_m), "DNp09_right": side_spikes(dnp09_m, right_m),
        "DNp_other_left": side_spikes(dnp_m & ~dnp09_m, left_m),
        "DNp_other_right": side_spikes(dnp_m & ~dnp09_m, right_m),
        "DAN_n_responsive": int((dan_counts > 0).sum()),
        "DAN_spikes_total": float(dan_counts.sum()),
    }
    stim_side = cand["side"]
    if stim_side in ("left", "right"):
        contra_side = "right" if stim_side == "left" else "left"
        ipsi = row[f"dm_spikes_{stim_side}"]
        contra = row[f"dm_spikes_{contra_side}"]
        row["ipsi_spikes"] = ipsi
        row["contra_spikes"] = contra
        row["ipsi_n_responsive"] = row[f"dm_n_resp_{stim_side}"]
        row["contra_n_responsive"] = row[f"dm_n_resp_{contra_side}"]
        row["LI"] = (ipsi - contra) / (ipsi + contra) if (ipsi + contra) > 0 else float("nan")
    else:
        row["ipsi_spikes"] = float("nan")
        row["contra_spikes"] = float("nan")
        row["ipsi_n_responsive"] = -1
        row["contra_n_responsive"] = -1
        row["LI"] = float("nan")
    return row


# ---------------------------------------------------------------------
# Mirror-consistency
# ---------------------------------------------------------------------

def build_pool_mapping(dm_side, dm_ct):
    keys = np.array([f"{ct}|{side}" for ct, side in zip(dm_ct, dm_side)])
    uniq, inverse = np.unique(keys, return_inverse=True)
    key_to_id = {k: i for i, k in enumerate(uniq)}
    swap = {"left": "right", "right": "left"}
    mirror_id = np.full(len(uniq), -1, dtype=np.int64)
    for i, k in enumerate(uniq):
        ct, side = k.split("|")
        if side in swap:
            mirror_id[i] = key_to_id.get(f"{ct}|{swap[side]}", -1)
    return inverse, mirror_id, uniq


def pooled_vec(dm_counts, inverse, n_pools):
    return np.bincount(inverse, weights=dm_counts, minlength=n_pools)


def mirror_consistency(vecL_dm, vecR_dm, inverse, mirror_id, n_pools):
    pL = pooled_vec(vecL_dm, inverse, n_pools)
    pR = pooled_vec(vecR_dm, inverse, n_pools)
    valid = mirror_id >= 0
    a = pL[valid]
    b = pR[mirror_id[valid]]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def cosine_sim_matrix(vectors: dict):
    names = list(vectors.keys())
    mat = np.stack([vectors[n] for n in names], axis=0).astype(np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = mat / norms
    sim = unit @ unit.T
    return names, sim


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fb = FastBrain(batch=WAVE_SIZE, device="cpu", dt=DT_MS, seed=0)
    dm_idx, dm_side, dm_ct = descending_motor_with_type(fb)
    dan_idx = dan_indices(fb)
    print(f"fb.N={fb.N}  dm_candidates={len(dm_idx)}  dan_candidates={len(dan_idx)}")

    candidates = build_candidates(fb)
    print(f"Built {len(candidates)} candidate populations (pre-batching).")
    by_family = {}
    for c in candidates:
        by_family.setdefault(c["family"], 0)
        by_family[c["family"]] += 1
    print("Candidates per family:", by_family)

    stimuli = [(c, r) for c in candidates for r in RATES]
    print(f"Total (candidate, rate) stimuli: {len(stimuli)}  "
          f"-> {len(stimuli) // WAVE_SIZE + 1} waves of <= {WAVE_SIZE}")

    rows = []
    vectors_150 = {}  # name -> dm response vector (rate=150 only)
    n_waves = 0
    for wave in chunked(stimuli, WAVE_SIZE):
        t0 = time.time()
        counts_np = run_wave(fb, wave)
        wall = time.time() - t0
        n_waves += 1
        for row_i, (c, r) in enumerate(wave):
            row = summarize_row(c, r, counts_np[row_i], dm_idx, dm_side, dm_ct, dan_idx)
            row["wall_s_wave"] = wall
            rows.append(row)
            if r == 150.0:
                vectors_150[c["name"]] = counts_np[row_i][dm_idx]
        print(f"  wave {n_waves:3d}: {len(wave):2d} stimuli, {wall:5.2f}s")

    summary_path = OUT_DIR / "candidate_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {summary_path} ({len(rows)} rows)")

    # ---- cosine similarity matrix (150 Hz DN response vectors) ----
    names, sim = cosine_sim_matrix(vectors_150)
    cos_path = OUT_DIR / "cosine_similarity.csv"
    with open(cos_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + names)
        for i, n in enumerate(names):
            writer.writerow([n] + [f"{v:.5f}" for v in sim[i]])
    print(f"Saved {cos_path} ({len(names)}x{len(names)})")

    # ---- mirror consistency (per family+subtype with both L and R @150Hz) ----
    inverse, mirror_id, pool_keys = build_pool_mapping(dm_side, dm_ct)
    n_pools = len(pool_keys)
    by_fam_subtype = {}
    for c in candidates:
        by_fam_subtype.setdefault((c["family"], c["subtype"]), {})[c["side"]] = c["name"]
    mirror_rows = []
    for (fam, subtype), sides in by_fam_subtype.items():
        lname, rname = sides.get("left"), sides.get("right")
        if lname in vectors_150 and rname in vectors_150:
            mc = mirror_consistency(vectors_150[lname], vectors_150[rname], inverse, mirror_id, n_pools)
            mirror_rows.append({"family": fam, "subtype": subtype, "mirror_consistency": mc})
    mirror_path = OUT_DIR / "mirror_consistency.csv"
    with open(mirror_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "subtype", "mirror_consistency"])
        writer.writeheader()
        writer.writerows(mirror_rows)
    print(f"Saved {mirror_path} ({len(mirror_rows)} rows)")

    # ---- pick a reference food channel for the gating test ----
    df_rows = pd.DataFrame(rows)
    alone_150 = df_rows[df_rows["rate_hz"] == 150.0]
    food_pool = alone_150[alone_150["family"].isin(["ORN", "ALPN"])
                           & alone_150["subtype"].isin(FOOD_GLOMERULI)
                           & (~alone_150["ignition"]) & (alone_150["dm_n_responsive"] > 0)]
    if len(food_pool):
        food_pool = food_pool.sort_values("dm_n_responsive", ascending=False)
        food_channel_name = food_pool.iloc[0]["name"]
    else:
        fallback = alone_150[alone_150["family"].isin(["ORN", "ALPN"])
                              & (~alone_150["ignition"]) & (alone_150["dm_n_responsive"] > 0)]
        fallback = fallback.sort_values("dm_n_responsive", ascending=False)
        food_channel_name = fallback.iloc[0]["name"] if len(fallback) else None
    print(f"Reference food channel for gating test: {food_channel_name}")

    gating_rows = []
    if food_channel_name is not None:
        food_cand = next(c for c in candidates if c["name"] == food_channel_name)
        zero_resp = alone_150[(alone_150["family"].isin(["endocrine", "neuromod"]))
                               & (alone_150["dm_n_responsive"] == 0)]
        gate_cands = [c for c in candidates if c["name"] in set(zero_resp["name"])]
        print(f"Gating candidates (zero DN response alone @150Hz): {len(gate_cands)}")

        # food alone (once)
        alone_wave = [(food_cand, GATE_FOOD_RATE)]
        alone_counts = run_wave(fb, alone_wave)[0]
        food_alone_dm = float(alone_counts[dm_idx].sum())
        food_alone_dan = float(alone_counts[dan_idx].sum())
        food_alone_all = float(alone_counts.sum())

        for gate_wave in chunked(gate_cands, WAVE_SIZE - 1):
            # Each row co-stimulates food_cand (@GATE_FOOD_RATE) and one gate
            # candidate (@GATE_STIM_RATE) over their union of neuron indices.
            union_idx = np.unique(np.concatenate(
                [food_cand["local_idx"]] + [gc["local_idx"] for gc in gate_wave]))
            col_pos = {int(v): i for i, v in enumerate(union_idx.tolist())}
            K = len(union_idx)
            rates_mat = np.zeros((fb.batch, K), dtype=np.float32)  # zero-padded to fb.batch rows
            for row_i, gc in enumerate(gate_wave):
                for v in food_cand["local_idx"].tolist():
                    rates_mat[row_i, col_pos[int(v)]] = GATE_FOOD_RATE
                for v in gc["local_idx"].tolist():
                    rates_mat[row_i, col_pos[int(v)]] = GATE_STIM_RATE
            fb.reset()
            fb.set_input_neurons(union_idx)
            fb.set_exc_indices(union_idx)
            counts_np = fb.run(torch.as_tensor(rates_mat), N_STEPS).numpy()[:len(gate_wave)]
            for row_i, gc in enumerate(gate_wave):
                combo_dm = float(counts_np[row_i][dm_idx].sum())
                combo_dan = float(counts_np[row_i][dan_idx].sum())
                combo_all = float(counts_np[row_i].sum())
                gating_rows.append({
                    "candidate": gc["name"], "family": gc["family"], "subtype": gc["subtype"],
                    "side": gc["side"], "food_channel": food_channel_name,
                    "food_alone_dm_spikes": food_alone_dm, "food_alone_dan_spikes": food_alone_dan,
                    "food_alone_total_spikes": food_alone_all,
                    "combo_dm_spikes": combo_dm, "combo_dan_spikes": combo_dan,
                    "combo_total_spikes": combo_all,
                    "delta_dm_spikes": combo_dm - food_alone_dm,
                    "delta_dan_spikes": combo_dan - food_alone_dan,
                    "pct_change_dm": (combo_dm - food_alone_dm) / food_alone_dm * 100.0 if food_alone_dm > 0 else float("nan"),
                })
        gating_path = OUT_DIR / "gating.csv"
        with open(gating_path, "w", newline="") as f:
            if gating_rows:
                writer = csv.DictWriter(f, fieldnames=list(gating_rows[0].keys()))
                writer.writeheader()
                writer.writerows(gating_rows)
        print(f"Saved {gating_path} ({len(gating_rows)} rows)")
    else:
        print("WARNING: no non-igniting, DN-responsive ORN/ALPN candidate found for gating reference; skipped gating test.")

    meta = {
        "n_candidates": len(candidates), "candidates_per_family": by_family,
        "n_stimuli": len(stimuli), "n_waves": n_waves,
        "food_channel_for_gating": food_channel_name,
        "ignition_threshold": IGNITION_THRESHOLD, "rates_hz": list(RATES),
        "window_ms": WINDOW_MS, "dt_ms": DT_MS,
        "wall_s_total": time.time() - t_start,
    }
    with open(OUT_DIR / "screen_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done in {meta['wall_s_total']:.1f}s. Meta: {meta}")


if __name__ == "__main__":
    main()
