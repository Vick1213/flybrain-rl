"""
Benchmark FastBrain's run() hot path against the reference PyTorch model on
the 'sugar' experiment (21 sugar GRN FlyWire IDs at 200 Hz).

Prints wall-clock seconds per simulated second for FastBrain.run() (sparse
input mode) at batch in {1, 8, 32} on cpu, and on mps if available/working,
for 0.2s simulated each. Also re-measures the reference TorchModel's
per-simulated-second wall time at batch 1 (0.02s simulated, to save time)
for comparison against the previously measured ~264 s/simsec.

Read-only w.r.t. eon-fly-brain/: only imports its loader functions/model
classes. As a safety net, restores eon-fly-brain/data/benchmark-results.csv
via `git checkout` afterwards in case anything on the import path appended
to it (the code paths used here never call save_result_csv, so this is
normally a no-op).
"""

import subprocess
import sys
from pathlib import Path
from time import perf_counter

import torch

REPO = Path(__file__).resolve().parent
EON_ROOT = REPO / "eon-fly-brain"
EON_CODE = EON_ROOT / "code"
if str(EON_CODE) not in sys.path:
    sys.path.insert(0, str(EON_CODE))

import pyarrow  # noqa: F401,E402
from run_pytorch import MODEL_PARAMS, DT, TorchModel, get_hash_tables, get_weights  # noqa: E402
from benchmark import path_comp, path_con, path_wt, get_experiment  # noqa: E402

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from flyrl.fastbrain import FastBrain  # noqa: E402


def bench_fastbrain(device, batch, sim_seconds):
    experiment = get_experiment("sugar")
    fb = FastBrain(batch=batch, device=device, seed=0)
    exc_idx = [fb.flyid_to_index[n] for n in experiment["neu_exc"]]
    fb.set_exc_indices(exc_idx)
    fb.set_input_neurons(exc_idx)
    rates = torch.full((batch, len(exc_idx)), experiment["stim_rate"], device=device)

    n_steps = int(round(sim_seconds * 1000.0 / DT))

    # Warm up (first-call allocation / lazy-init overhead), then reset state
    # so the timed run starts from a clean slate.
    fb.run(rates, 5)
    fb.reset()

    if device == "mps":
        torch.mps.synchronize()
    t0 = perf_counter()
    counts = fb.run(rates, n_steps)
    if device == "mps":
        torch.mps.synchronize()
    wall = perf_counter() - t0

    sim_time = n_steps * DT / 1000.0
    # Match the reference codebase's own convention (see smoke_test.py /
    # run_pytorch.py's realtime_ratio): normalize by (sim_seconds * batch),
    # i.e. "wall-clock seconds per simulated second, per parallel trial" --
    # this is what makes the batch=1 number comparable to the reference's
    # quoted "264 s/simsec".
    ratio = wall / (sim_time * batch)
    return wall, sim_time, ratio, float(counts.sum().item())


def bench_reference(batch, sim_seconds, device="cpu"):
    experiment = get_experiment("sugar")
    flyid2i, _ = get_hash_tables(str(path_comp))
    exc_indices = [flyid2i[n] for n in experiment["neu_exc"]]
    weights = get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(device=device)
    num_neurons = weights.shape[0]
    model = TorchModel(
        batch, num_neurons, DT, MODEL_PARAMS, weights, exc_indices=exc_indices, device=device
    )
    conductance, delay_buffer, spikes, v, refrac = model.state_init()
    rates = torch.zeros(batch, num_neurons, device=device)
    rates[:, exc_indices] = experiment["stim_rate"]

    n_steps = int(round(sim_seconds * 1000.0 / DT))
    t0 = perf_counter()
    with torch.no_grad():
        for _ in range(n_steps):
            conductance, delay_buffer, spikes, v, refrac = model(
                rates, conductance, delay_buffer, spikes, v, refrac
            )
    if device == "cuda":
        torch.cuda.synchronize()
    wall = perf_counter() - t0
    sim_time = n_steps * DT / 1000.0
    return wall, sim_time, wall / sim_time


def _restore_eon_csv():
    csv_path = EON_ROOT / "data" / "benchmark-results.csv"
    try:
        subprocess.run(
            ["git", "-C", str(EON_ROOT), "checkout", "--", "data/benchmark-results.csv"],
            check=False, capture_output=True,
        )
    except Exception:
        pass
    return csv_path


def main():
    print("=" * 78)
    print("Reference PyTorch TorchModel  (sugar, batch=1, 0.02s simulated, CPU)")
    print("=" * 78)
    ref_wall, ref_sim, ref_ratio = bench_reference(batch=1, sim_seconds=0.02, device="cpu")
    print(
        f"  wall={ref_wall:.3f}s  sim={ref_sim:.3f}s  -> {ref_ratio:.1f} s/simsec "
        f"(prior full measurement: ~264 s/simsec at 0.1s simulated)"
    )

    print()
    print("=" * 78)
    print("FastBrain.run() hot path, sparse input mode, sugar experiment, 0.2s simulated")
    print("(s/simsec normalized per trial: wall / (sim_seconds * batch), same convention")
    print(" the reference codebase uses -- so this is directly comparable to the ~264")
    print(" s/simsec reference number above, even at batch > 1)")
    print("=" * 78)
    rows = []
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    else:
        print(f"  device=mps: not available ({torch.backends.mps.is_built()=}), skipping")

    for device in devices:
        for batch in [1, 8, 32]:
            try:
                wall, sim_time, ratio, nspikes = bench_fastbrain(device, batch, 0.2)
                rows.append((device, batch, wall, sim_time, ratio, nspikes))
                print(
                    f"  device={device:4s} batch={batch:3d}  wall={wall:7.3f}s  "
                    f"sim={sim_time:.3f}s  -> {ratio:9.4f} s/simsec   (total spikes={nspikes:.0f})"
                )
            except Exception as e:
                print(f"  device={device:4s} batch={batch:3d}  FAILED: {type(e).__name__}: {e}")

    print()
    print("=" * 78)
    print("Summary (seconds of wall-clock per simulated second; lower is better)")
    print("=" * 78)
    print(f"  {'backend':22s} {'batch':>6s} {'s/simsec':>12s}   speedup vs reference")
    print(f"  {'reference (matmul)':22s} {1:>6d} {ref_ratio:12.2f}   1.0x")
    for device, batch, wall, sim_time, ratio, nspikes in rows:
        speedup = ref_ratio / ratio if ratio > 0 else float("inf")
        print(f"  {'FastBrain ' + device:22s} {batch:>6d} {ratio:12.4f}   {speedup:6.1f}x")

    _restore_eon_csv()


if __name__ == "__main__":
    main()
