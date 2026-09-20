# NOTICE

This project builds on several upstream projects and datasets. None of their
code is vendored into this repository's git history except where noted;
`setup.sh` clones the upstream repos at pinned commits into sibling
directories that are gitignored here.

## eonsystemspbc/fly-brain

- Repository: https://github.com/eonsystemspbc/fly-brain
- License: GPL-2.0 (see `LICENSE` in this repo, copied verbatim from
  `eon-fly-brain/LICENSE`)
- Implements the Shiu et al. leaky-integrate-and-fire (LIF) whole-brain
  connectome model.
- Usage: imported at runtime (not vendored) by `flyrl/fastbrain.py`, which
  reads its weight-loading utilities and model constants from
  `eon-fly-brain/code/run_pytorch.py`. `flyrl/fastbrain.py` ("FastBrain") is
  a derivative, event-driven reimplementation of the dynamics in
  `run_pytorch.py`, and is therefore distributed under GPL-2.0 as well (see
  this repo's `LICENSE`).

## TuragaLab/flybody

- Repository: https://github.com/TuragaLab/flybody
- License: Apache-2.0
- Provides a MuJoCo biomechanical model of the fly body.
- Usage: used for 3D rendering of the fly body (`fly3d/`), independent of
  the connectome simulation.

## abgnydn/webgpu-fly

- Repository: https://github.com/abgnydn/webgpu-fly
- License: MIT
- A browser-based WebGPU viewer for the FlyWire connectome.
- Usage: reference for a browser viewer, and the source of the tooling used
  to download FlyWire connectome annotations
  (`webgpu-fly/tools/download_data.sh`).

## snedea/flybrain

- Repository: https://github.com/snedea/flybrain
- License: MIT
- A 2D browser demo of fly behavior plus neuron classification CSVs.
- Usage: reference implementation / data source, cloned as `browser-sim/`.

## FlyWire connectome data

- Dorkenwald et al. 2024, *Nature* — FlyWire whole-brain connectome
  reconstruction.
- Schlegel et al. 2024, *Nature* — whole-brain annotation and cell typing
  used to build the connectome's neuron/synapse tables.
- Usage: the connectome (FlyWire v783, 138,639 neurons, ~15M synapses)
  underlying the LIF model simulated by `flyrl/fastbrain.py`.

## Shiu et al. 2024

- Shiu et al. 2024, *Nature* — the leaky integrate-and-fire (LIF)
  whole-brain model reimplemented by `eon-fly-brain` and, in
  event-driven form, by `flyrl/fastbrain.py`.
