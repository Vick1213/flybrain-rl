"""
Equivalence / correctness tests for flyrl.fastbrain.FastBrain against the
reference eon-fly-brain/code/run_pytorch.py TorchModel.

These tests only *import* from eon-fly-brain (read-only); they never modify
files under eon-fly-brain/.

Determinism strategy for the equivalence test: both models are driven with
an IDENTICAL precomputed Poisson input spike train (generated once with a
seeded generator) instead of letting each model sample its own Bernoulli
draws. For the reference TorchModel this is done by swapping in a small
replay module in place of its `.poisson` submodule (a `PoissonSpikeGenerator`
drop-in that just replays precomputed values); for FastBrain this is done
via the `voltage_stim_override` hook on `step()`. This isolates the
comparison to the recurrent (event-driven vs matmul) and LIF/synapse
dynamics, since the exogenous input is bit-identical between the two.
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
EON_CODE = REPO / "eon-fly-brain" / "code"
if str(EON_CODE) not in sys.path:
    sys.path.insert(0, str(EON_CODE))

import pyarrow  # noqa: F401,E402
from run_pytorch import MODEL_PARAMS, DT, TorchModel, get_hash_tables, get_weights  # noqa: E402
from benchmark import path_comp, path_con, path_wt, get_experiment  # noqa: E402

sys.path.insert(0, str(REPO))
from flyrl.fastbrain import FastBrain  # noqa: E402


class ReplayPoisson(torch.nn.Module):
    """Drop-in replacement for PoissonSpikeGenerator: replays a precomputed
    (num_steps, B, N) spike train (already scaled by scalePoisson) instead
    of sampling from `rates`."""

    def __init__(self, train):
        super().__init__()
        self.train = train
        self.t = 0

    def forward(self, rates, generator=None):
        out = self.train[self.t]
        self.t += 1
        return out


def _build_reference(batch, device="cpu"):
    flyid2i, i2flyid = get_hash_tables(str(path_comp))
    experiment = get_experiment("sugar")
    exc_indices = [flyid2i[n] for n in experiment["neu_exc"]]
    weights = get_weights(str(path_con), str(path_comp), str(path_wt), csr=True).to(device=device)
    num_neurons = weights.shape[0]
    model = TorchModel(
        batch, num_neurons, DT, MODEL_PARAMS, weights, exc_indices=exc_indices, device=device
    )
    state = model.state_init()
    return model, state, exc_indices, num_neurons, flyid2i, experiment


def _make_poisson_train(num_steps, batch, num_neurons, exc_indices, stim_rate, seed=1234):
    """Precompute the exogenous Poisson spike train ONCE for both models.
    Only exc_indices columns can ever be nonzero (rates are 0 elsewhere in
    the sugar experiment), so we only draw randomness for those K columns
    and place them into a dense (num_steps, B, N) tensor."""
    gen = torch.Generator().manual_seed(seed)
    prob = stim_rate * DT / 1000.0
    k = len(exc_indices)
    bern = torch.bernoulli(torch.full((num_steps, batch, k), prob), generator=gen)
    scale = MODEL_PARAMS["scalePoisson"]
    idx = torch.as_tensor(exc_indices, dtype=torch.long)
    train = torch.zeros(num_steps, batch, num_neurons)
    train.index_copy_(2, idx, bern * scale)
    return train


def _run_reference(model, state, poisson_train, num_steps):
    conductance, delay_buffer, spikes, v, refrac = state
    model.poisson = ReplayPoisson(poisson_train)
    batch, num_neurons = spikes.shape
    rates = torch.zeros(batch, num_neurons)  # unused; ReplayPoisson ignores it
    counts = torch.zeros(batch, num_neurons)
    with torch.no_grad():
        for _ in range(num_steps):
            conductance, delay_buffer, spikes, v, refrac = model(
                rates, conductance, delay_buffer, spikes, v, refrac
            )
            counts += spikes
    return counts


def _run_fastbrain_replay(fb, poisson_train, num_steps):
    wscale = MODEL_PARAMS["wScale"]
    counts = torch.zeros(fb.batch, fb.N)
    with torch.no_grad():
        for t in range(num_steps):
            override = poisson_train[t] * wscale
            spike = fb.step(voltage_stim_override=override)
            counts += spike.to(torch.float32)
    return counts


def test_equivalence_sugar_0p1s_identical_input():
    """0.1s simulated (1000 steps), batch 1, CPU, identical injected Poisson
    input spike train fed to both the reference TorchModel and FastBrain."""
    num_steps = 1000
    model, state, exc_indices, num_neurons, flyid2i, experiment = _build_reference(batch=1)
    poisson_train = _make_poisson_train(
        num_steps, 1, num_neurons, exc_indices, experiment["stim_rate"]
    )

    ref_counts = _run_reference(model, state, poisson_train, num_steps)

    fb = FastBrain(batch=1, device="cpu", seed=0)
    assert fb.N == num_neurons
    fb.set_exc_indices(exc_indices)
    fb_counts = _run_fastbrain_replay(fb, poisson_train, num_steps)

    ref = ref_counts[0].numpy()
    fbc = fb_counts[0].numpy()

    ref_active = set(np.nonzero(ref)[0].tolist())
    fb_active = set(np.nonzero(fbc)[0].tolist())
    union = ref_active | fb_active
    assert len(union) > 0, "no neuron spiked in either model -- test is vacuous"

    symmetric_diff = ref_active ^ fb_active
    mismatch_frac = len(symmetric_diff) / len(union)
    assert mismatch_frac <= 0.02, (
        f"active-neuron-set mismatch {mismatch_frac:.4f} exceeds 2% "
        f"(ref={len(ref_active)}, fb={len(fb_active)}, diff={len(symmetric_diff)})"
    )

    corr = np.corrcoef(ref, fbc)[0, 1]
    assert corr > 0.99, f"per-neuron spike-count correlation {corr:.5f} <= 0.99"

    total_ref = float(ref.sum())
    total_fb = float(fbc.sum())
    denom = max(total_ref, total_fb, 1.0)
    assert abs(total_ref - total_fb) / denom <= 0.02, (
        f"total spike count mismatch: ref={total_ref}, fb={total_fb}"
    )


def test_batch_independence():
    """batch=2, input only to element 0 -> element 1 must stay completely
    silent (no cross-batch leakage in the flattened index_add_ scatter)."""
    experiment = get_experiment("sugar")
    fb = FastBrain(batch=2, device="cpu", seed=0)
    exc_indices = [fb.flyid_to_index[n] for n in experiment["neu_exc"]]
    fb.set_exc_indices(exc_indices)
    fb.set_input_neurons(exc_indices)

    rates = torch.zeros(2, len(exc_indices))
    rates[0, :] = experiment["stim_rate"]

    counts = fb.run(rates, 500)

    assert counts[1].sum().item() == 0.0, "batch element 1 received spikes with zero input rate"
    assert counts[0].sum().item() > 0.0, "batch element 0 (driven) produced no spikes at all"


def test_reset_reproducible():
    """run, reset(), run again with the same seed -> identical spike counts."""
    experiment = get_experiment("sugar")
    fb = FastBrain(batch=1, device="cpu", seed=42)
    exc_indices = [fb.flyid_to_index[n] for n in experiment["neu_exc"]]
    fb.set_exc_indices(exc_indices)
    fb.set_input_neurons(exc_indices)
    rates = torch.full((1, len(exc_indices)), experiment["stim_rate"])

    fb.manual_seed(42)
    counts1 = fb.run(rates, 500)

    fb.reset()
    fb.manual_seed(42)
    counts2 = fb.run(rates, 500)

    assert torch.equal(counts1, counts2)
    assert counts1.sum().item() > 0
