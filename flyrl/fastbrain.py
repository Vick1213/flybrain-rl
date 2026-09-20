"""
FastBrain: event-driven, batched reimplementation of the eon-fly-brain
whole-fly-brain LIF simulator (see eon-fly-brain/code/run_pytorch.py).

Goal: bit-for-bit-equivalent *dynamics* to the reference TorchModel /
AlphaLIF / AlphaSynapse / LIFNeuron / PoissonSpikeGenerator stack, but
event-driven: each step only touches the outgoing synapses of neurons that
actually spiked on the previous step, instead of doing a dense/sparse
matmul over all ~15.1M synapses every 0.1 ms tick.

This module only *reads* eon-fly-brain (weights cache, ID tables, model
constants) via its own loader functions (get_hash_tables, get_weights) and
never modifies files under eon-fly-brain/.

Reference update order replicated exactly (see run_pytorch.py):

  TorchModel.forward(rates, conductance, delay_buffer, spikes, v, refrac):
    poisson_spikes = bernoulli(rates * dt/1000) * scalePoisson
    voltage_stim   = wScale * poisson_spikes
    weighted_spikes = spikes @ W.T          # spikes = PREVIOUS step's output
    recurrent_input = wScale * weighted_spikes

    AlphaLIF.forward(recurrent_input, voltage_stim, conductance,
                      delay_buffer, spikes, v, refrac):
      refrac = where(spikes > 0, 0, refrac + 1)          # spikes = previous
      refrac_mask = (refrac >= refrac_steps).float()
      conductance_new, delay_buffer = AlphaSynapse.forward(
          recurrent_input, conductance, delay_buffer, refrac_mask):
            conductance_new = conductance*(1 - dt/tauSyn)
                               + delay_buffer[:, 0, :] * refrac_mask
            delay_buffer = roll(delay_buffer, -1, dim=1)
            delay_buffer[:, -1, :] = recurrent_input
      spikes, v = LIFNeuron.forward(conductance, voltage_stim, v):
            # NOTE: uses the OLD conductance (pre-update), not conductance_new
            v = v + voltage_stim
            v = v + (dt/tauMem) * (conductance - (v - vRest))
            spike = (v > vThreshold).float()
            v = v - (v - vReset) * spike
      conductance_new = conductance_new - conductance_new * spikes  # shunt reset

FastBrain replicates every line above. The only thing that changes is *how*
`weighted_spikes` (the recurrent input) and the alpha-synapse delay line are
computed: instead of `spikes @ W.T` (an O(N^2) / O(nnz) matmul every step),
we look up the outgoing synapses of only the neurons that spiked on the
previous step (CSC-like arrays keyed by presynaptic index) and scatter-add
their (post, weight) contributions into the delay ring buffer with a single
`index_add_` per step (batch dimension flattened into the destination
index). The rolling (steps_delay+1, N) delay buffer is replaced by a ring
buffer indexed by a step counter (no per-step copy).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Wire up imports of the reference model's own loader functions / constants.
# We only ever *import* from eon-fly-brain/code; we never write to it.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_EON_ROOT = _REPO_ROOT / "eon-fly-brain"
_EON_CODE = _EON_ROOT / "code"
if str(_EON_CODE) not in sys.path:
    sys.path.insert(0, str(_EON_CODE))

import pyarrow  # noqa: F401,E402  - must precede torch-heavy imports (see run_pytorch.py)
from run_pytorch import MODEL_PARAMS, DT, get_hash_tables, get_weights  # noqa: E402
from benchmark import path_comp, path_con, path_wt  # noqa: E402

__all__ = ["FastBrain", "MODEL_PARAMS", "DT"]


def _as_index_tensor(idx, device):
    if isinstance(idx, torch.Tensor):
        t = idx.to(dtype=torch.long, device=device)
    else:
        t = torch.as_tensor(list(idx), dtype=torch.long, device=device)
    return t.reshape(-1)


class FastBrain:
    """Event-driven whole-fly-brain LIF simulator.

    Parameters
    ----------
    batch : int
        Number of independent brains simulated in parallel, sharing weights.
    device : str
        'cpu' or 'mps' (sparse CSR is not usable on MPS; this class never
        constructs a sparse tensor -- everything is dense 1-D index/value
        tensors + index_add_).
    dt : float
        Simulation timestep in ms. Defaults to the reference DT (0.1 ms).
    params : dict, optional
        Model parameters; defaults to the reference MODEL_PARAMS.
    seed : int, optional
        Seed for this instance's own RNG (used for Poisson sampling in
        step()/run()). Does NOT reproduce the reference's RNG stream bit for
        bit (different algorithm/order); use the `voltage_stim_override`
        hook in `step()` for exact cross-implementation determinism.
    """

    def __init__(self, batch=1, device="cpu", dt=None, params=None, seed=None):
        self.batch = int(batch)
        self.device = torch.device(device)
        self.dt = float(dt) if dt is not None else DT
        self.params = dict(MODEL_PARAMS if params is None else params)

        self._exc_indices = None  # neurons with refrac_steps forced to 0
        self._input_idx = None  # optional sparse-input neuron index tensor

        self.flyid_to_index, self.index_to_flyid = get_hash_tables(str(path_comp))

        self._load_and_build_csc()

        # RNG for Poisson sampling in step()/run(). Try to keep it on the
        # compute device; MPS generator support is spotty across torch
        # versions, so fall back to a CPU generator + device cast if needed.
        self._gen_device = "cpu"
        self._generator = torch.Generator(device="cpu")
        if device == "mps":
            try:
                g = torch.Generator(device="mps")
                if seed is not None:
                    g.manual_seed(seed)
                # smoke-test it
                torch.bernoulli(torch.zeros(1, device="mps"), generator=g)
                self._generator = g
                self._gen_device = "mps"
            except Exception:
                self._generator = torch.Generator(device="cpu")
                self._gen_device = "cpu"
        if seed is not None:
            self._generator.manual_seed(seed)

        self._build_derived_constants()
        self.reset()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _load_and_build_csc(self):
        """Load weights via the reference's own loader and rebuild them as
        CSC-like arrays keyed by PRESYNAPTIC index (outgoing synapses),
        since the event-driven step needs "for each spiking neuron, which
        postsynaptic neurons/weights does it drive" -- the reference's own
        cached CSR is keyed by postsynaptic index (incoming), so we derive
        our own layout from the COO cache instead (same underlying
        (post, pre, weight) triples, just re-sorted).
        """
        weight_coo = get_weights(str(path_con), str(path_comp), str(path_wt), csr=False)
        if not weight_coo.is_coalesced():
            weight_coo = weight_coo.coalesce()

        self.N = int(weight_coo.shape[0])
        indices = weight_coo.indices()
        post_idx = indices[0]
        pre_idx = indices[1]
        vals = weight_coo.values().to(torch.float32)

        # Sort by presynaptic index to build a CSC-like (indptr over pre,
        # post_index, weight) layout. Pre-scale weights by wScale here so
        # the hot loop does one fewer multiply.
        order = torch.argsort(pre_idx, stable=True)
        pre_sorted = pre_idx[order]
        post_sorted = post_idx[order].to(torch.long)
        w_scale = float(self.params["wScale"])
        val_sorted = (vals[order] * w_scale).to(torch.float32)

        counts = torch.bincount(pre_sorted, minlength=self.N)
        indptr = torch.zeros(self.N + 1, dtype=torch.long)
        torch.cumsum(counts, dim=0, out=indptr[1:])

        self.indptr = indptr.to(self.device)
        self.post_index = post_sorted.to(self.device)
        self.weight = val_sorted.to(self.device)
        self.nnz = int(self.weight.numel())

    def _build_derived_constants(self):
        p = self.params
        dt = self.dt
        self.time_factor_syn = dt / p["tauSyn"]
        self.steps_delay = int(p["tDelay"] / dt)
        self.ring_len = self.steps_delay + 1
        self.time_factor_mem = dt / p["tauMem"]
        self.v_reset = float(p["vReset"])
        self.v_rest = float(p["vRest"])
        self.v_threshold = float(p["vThreshold"])
        self.v0 = float(p["v0"])
        self.poisson_prob_scale = dt / 1000.0
        self.voltage_stim_const = float(p["wScale"]) * float(p["scalePoisson"])
        self.base_refrac = int(round(p["tRefrac"] / dt))

    def set_exc_indices(self, idx):
        """Mark neurons whose refractory gate is permanently open (refrac_steps
        forced to 0), matching the reference's `exc_indices` constructor arg
        (used for the externally-driven / sensory input neurons in the
        benchmark experiments). Rebuilds state via reset()."""
        self._exc_indices = None if idx is None else _as_index_tensor(idx, self.device)
        self.reset()

    def manual_seed(self, seed):
        """Reseed this instance's own Poisson-sampling RNG (does not touch
        v/conductance/refrac/ring state -- see reset() for that)."""
        self._generator.manual_seed(seed)

    def set_input_neurons(self, idx):
        """Optional perf hook: restrict Poisson sampling to K neurons.

        After calling this, `rates` passed to step()/run() must have shape
        (B, K) instead of (B, N); Poisson draws (and thus voltage_stim) are
        computed only for these K neurons, all others implicitly get 0 Hz.
        """
        self._input_idx = None if idx is None else _as_index_tensor(idx, self.device)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset(self):
        """Reset all state (v, conductance, refractory counters, delay ring)."""
        B, N, device = self.batch, self.N, self.device

        refrac_steps = torch.full((N,), float(self.base_refrac), dtype=torch.float32, device=device)
        if self._exc_indices is not None:
            refrac_steps[self._exc_indices] = 0.0
        self.refrac_steps = refrac_steps

        self.v = torch.full((B, N), self.v0, dtype=torch.float32, device=device)
        self.conductance = torch.zeros((B, N), dtype=torch.float32, device=device)
        self.spikes = torch.zeros((B, N), dtype=torch.float32, device=device)
        self.refrac = refrac_steps.unsqueeze(0).expand(B, N).clone()

        # Ring buffer laid out (ring_len, B, N) so that ring[pos] is a
        # contiguous (B, N) slab (fast to zero / scatter / read).
        self.ring = torch.zeros((self.ring_len, B, N), dtype=torch.float32, device=device)
        self.ring_ptr = 0
        self._step_count = 0

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------
    def _sample_voltage_stim(self, rates):
        """Poisson input -> voltage_stim, exactly replicating:
        voltage_stim = wScale * (bernoulli(rates * dt/1000) * scalePoisson)
        """
        B, N = self.batch, self.N
        rates = torch.as_tensor(rates, dtype=torch.float32, device=self.device)
        gen = self._generator if self._gen_device == self.device.type else None
        bern = torch.bernoulli(rates * self.poisson_prob_scale, generator=gen)
        if self._input_idx is not None:
            voltage_stim = torch.zeros((B, N), dtype=torch.float32, device=self.device)
            voltage_stim.index_add_(1, self._input_idx, bern * self.voltage_stim_const)
        else:
            voltage_stim = bern * self.voltage_stim_const
        return voltage_stim

    def _scatter_prev_spikes(self, spikes_prev, dest_flat):
        """Gather-scatter the outgoing synapses of neurons that spiked on the
        PREVIOUS step into `dest_flat`, a flattened (B*N,) view of the ring
        slot that is about to become the newest delay-buffer entry.

        Equivalent to: dest[b, post] += sum_{pre: spikes_prev[b,pre]>0} weight[post,pre]
        i.e. wScale * (spikes_prev @ W.T), computed event-driven.
        """
        nz = torch.nonzero(spikes_prev, as_tuple=False)  # (S, 2): (batch, neuron)
        if nz.shape[0] == 0:
            return
        b_idx = nz[:, 0]
        pre_idx = nz[:, 1]

        starts = self.indptr[pre_idx]
        ends = self.indptr[pre_idx + 1]
        degrees = ends - starts

        edge_batch = torch.repeat_interleave(b_idx, degrees)
        edge_starts = torch.repeat_interleave(starts, degrees)
        seg_prefix = torch.repeat_interleave(torch.cumsum(degrees, dim=0) - degrees, degrees)
        flat_pos = torch.arange(edge_batch.numel(), device=self.device)
        edge_ptr = edge_starts + (flat_pos - seg_prefix)

        post_e = self.post_index[edge_ptr]
        weight_e = self.weight[edge_ptr]
        flat_idx = edge_batch * self.N + post_e
        dest_flat.index_add_(0, flat_idx, weight_e)

    @torch.no_grad()
    def _advance(self, rates=None, voltage_stim_override=None):
        """Advance one dt. Returns the raw (float, 0/1) spike tensor (B, N)."""
        if voltage_stim_override is not None:
            voltage_stim = voltage_stim_override.to(device=self.device, dtype=torch.float32)
        else:
            voltage_stim = self._sample_voltage_stim(rates)

        spikes_prev = self.spikes

        # refrac update (uses PREVIOUS step's spikes)
        self.refrac = torch.where(spikes_prev > 0, torch.zeros_like(self.refrac), self.refrac + 1)
        refrac_mask = (self.refrac >= self.refrac_steps.unsqueeze(0)).to(torch.float32)

        # --- AlphaSynapse.forward ---
        head_pos = self.ring_ptr
        delay_head = self.ring[head_pos]  # contiguous (B, N) view
        old_conductance = self.conductance
        conductance_new = old_conductance * (1.0 - self.time_factor_syn) + delay_head * refrac_mask

        # write this step's recurrent input (from spikes_prev) into the slot
        # we just consumed; ring_ptr advances by +1 (mod ring_len), matching
        # the reference's torch.roll(shifts=-1) + append-at-end semantics.
        delay_head.zero_()
        self._scatter_prev_spikes(spikes_prev, delay_head.view(-1))
        self.ring_ptr = (self.ring_ptr + 1) % self.ring_len

        # --- LIFNeuron.forward (uses OLD conductance, pre-update) ---
        v = self.v + voltage_stim
        v = v + self.time_factor_mem * (old_conductance - (v - self.v_rest))
        spike = (v > self.v_threshold).to(torch.float32)
        v = v - (v - self.v_reset) * spike

        # shunting conductance reset on spike
        conductance_new = conductance_new - conductance_new * spike

        self.conductance = conductance_new
        self.v = v
        self.spikes = spike
        self._step_count += 1
        return spike

    def step(self, rates=None, voltage_stim_override=None):
        """Advance one dt given `rates` (B, N) Hz (or (B, K) if
        set_input_neurons() was called). Returns a bool spike tensor (B, N).

        `voltage_stim_override`, if given, is a (B, N) tensor added directly
        in place of the internally-sampled Poisson voltage_stim (used by the
        equivalence tests to inject an identical, precomputed input spike
        train into both this model and the reference)."""
        spike = self._advance(rates=rates, voltage_stim_override=voltage_stim_override)
        return spike.to(torch.bool)

    def run(self, rates, n_steps, readout_idx=None):
        """Advance n_steps with constant `rates`. Returns spike COUNTS per
        neuron over the window, as (B, N), or (B, len(readout_idx)) if
        readout_idx is given."""
        with torch.no_grad():
            if readout_idx is not None:
                ridx = _as_index_tensor(readout_idx, self.device)
                counts = torch.zeros((self.batch, ridx.numel()), dtype=torch.float32, device=self.device)
                for _ in range(n_steps):
                    spike = self._advance(rates=rates)
                    counts += spike.index_select(1, ridx)
            else:
                counts = torch.zeros((self.batch, self.N), dtype=torch.float32, device=self.device)
                for _ in range(n_steps):
                    spike = self._advance(rates=rates)
                    counts += spike
        return counts
