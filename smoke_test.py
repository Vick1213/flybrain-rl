"""
Standalone smoke test for the PyTorch backend of the whole-fly-brain LIF model.

Imports the model classes directly from eon-fly-brain/code/run_pytorch.py
(bypassing main.py/benchmark.py's backend dispatch), builds the network from
the connectome data files the same way run_pytorch.py does, and runs a short
simulation on a chosen device (cpu or mps).

Usage:
    python smoke_test.py --device cpu
    PYTORCH_ENABLE_MPS_FALLBACK=1 python smoke_test.py --device mps
"""

import sys
import argparse
import traceback
from pathlib import Path
from time import perf_counter as time

REPO = Path(__file__).resolve().parent / 'eon-fly-brain'
sys.path.insert(0, str(REPO / 'code'))

import torch  # noqa: E402

from run_pytorch import (  # noqa: E402
    MODEL_PARAMS, DT, TorchModel, get_hash_tables, get_weights,
)
from benchmark import (  # noqa: E402
    path_comp, path_con, path_wt, get_experiment,
)


def main():
    parser = argparse.ArgumentParser(description='PyTorch fly-brain smoke test')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--t_run', type=float, default=0.1, help='Simulated seconds')
    parser.add_argument('--n_run', type=int, default=1, help='Batch size (trials)')
    args = parser.parse_args()

    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        print(f"MPS not available on this machine (built={torch.backends.mps.is_built()}).")
        sys.exit(1)

    t_sim_ms = args.t_run * 1000.0
    num_steps = int(t_sim_ms / DT)

    experiment = get_experiment('sugar')
    stim_rate = experiment['stim_rate']

    print(f"Device requested: {device}")
    print(f"t_run={args.t_run}s, n_run={args.n_run}, steps={num_steps}, dt={DT}ms")

    try:
        # ----- ID mapping -----
        flyid2i, i2flyid = get_hash_tables(str(path_comp))
        exc_indices = [flyid2i[n] for n in experiment['neu_exc']]

        # ----- Weights (uses cached pickles in data/ if present) -----
        t0 = time()
        weights = get_weights(str(path_con), str(path_comp), str(path_wt), csr=True)
        weights = weights.to(device=device)
        num_neurons = weights.shape[0]
        nnz = weights._nnz() if hasattr(weights, '_nnz') else weights.values().numel()
        t_weights = time() - t0
        print(f"Weight load+move: {t_weights:.3f}s")
        print(f"Weight tensor: layout={weights.layout}, dtype={weights.dtype}, "
              f"shape={tuple(weights.shape)}, nnz={nnz}")

        # ----- Model -----
        model = TorchModel(
            args.n_run, num_neurons, DT, MODEL_PARAMS, weights,
            exc_indices=exc_indices, device=device,
        )
        conductance, delay_buffer, spikes, v, refrac = model.state_init()

        rates = torch.zeros(args.n_run, num_neurons, device=device)
        rates[:, exc_indices] = stim_rate

        # ----- Run -----
        n_spikes_total = 0
        spiked_neuron_set = set()

        t_start = time()
        with torch.no_grad():
            for t_step in range(num_steps):
                conductance, delay_buffer, spikes, v, refrac = model(
                    rates, conductance, delay_buffer, spikes, v, refrac
                )
                spike_mask = spikes > 0
                if spike_mask.any():
                    b_idx, n_idx = spike_mask.nonzero(as_tuple=True)
                    n_spikes_total += len(b_idx)
                    spiked_neuron_set.update(n_idx.cpu().tolist())

        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()

        wall_clock = time() - t_start

        print("")
        print("===== RESULTS =====")
        print(f"Device:                  {device}")
        print(f"Number of neurons:       {num_neurons}")
        print(f"Nonzero synapses (nnz):  {nnz}")
        print(f"Wall-clock (sim loop):   {wall_clock:.3f}s")
        print(f"Simulated time:          {args.t_run}s x {args.n_run} trial(s) = "
              f"{args.t_run * args.n_run}s")
        print(f"Wall-clock per sim-sec:  {wall_clock / (args.t_run * args.n_run):.3f}s/simsec")
        print(f"Neurons that spiked:     {len(spiked_neuron_set)}")
        print(f"Total spikes:            {n_spikes_total}")

    except Exception as e:
        print(f"ERROR on device={device}: {e}")
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()
