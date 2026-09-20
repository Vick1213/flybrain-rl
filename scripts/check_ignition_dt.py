"""Check that olfactory "ignition" is not a timestep artifact.

Stimulates a few left-side populations alone for 300 ms at dt = 0.1 ms and
dt = 0.5 ms and records total network spikes and number of active neurons.
Writes results/ignition_dt_check.json.

Run from the project root:  .venv/bin/python scripts/check_ignition_dt.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flyrl.fastbrain import FastBrain  # noqa: E402

ANNOTATIONS = ROOT / "webgpu-fly/data/raw/flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
STIMS = [("ORN_DM1", 5.0), ("ORN_DM1", 50.0), ("ORN_DA1", 20.0), ("LC10e", 150.0), ("LPLC4", 150.0)]
SIM_MS = 300


def main():
    ann = pd.read_csv(ANNOTATIONS, sep="\t", low_memory=False)
    out = {"sim_ms": SIM_MS, "side": "left", "results": {}}
    for dt in (0.1, 0.5):
        fb = FastBrain(batch=len(STIMS), device="cpu", dt=dt, seed=1)
        groups = []
        for cell_type, _ in STIMS:
            ids = ann[(ann.cell_type == cell_type) & (ann.side == "left")].root_id.tolist()
            groups.append([fb.flyid_to_index[i] for i in ids if i in fb.flyid_to_index])
        union = sorted(set(sum(groups, [])))
        pos = {n: k for k, n in enumerate(union)}
        fb.set_input_neurons(torch.tensor(union))
        rates = torch.zeros(len(STIMS), len(union))
        for b, (g, (_, rate)) in enumerate(zip(groups, STIMS)):
            rates[b, [pos[n] for n in g]] = rate
        t0 = time.time()
        counts = fb.run(rates, int(SIM_MS / dt))
        wall = time.time() - t0
        for b, (g, (cell_type, rate)) in enumerate(zip(groups, STIMS)):
            key = f"{cell_type}@{rate:g}Hz"
            out["results"].setdefault(key, {"n_stimulated": len(g)})[f"dt_{dt}"] = {
                "total_spikes": int(counts[b].sum()),
                "active_neurons": int((counts[b] > 0).sum()),
            }
        print(f"dt={dt} ms done in {wall:.1f} s")
    path = ROOT / "results/ignition_dt_check.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["results"], indent=1))


if __name__ == "__main__":
    main()
