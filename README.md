# flybrain-rl

Reinforcement learning on a frozen whole-*Drosophila*-brain connectome
leaky-integrate-and-fire (LIF) model (FlyWire v783, 138,639 neurons, ~15M
synapses), with only small encoder/decoder layers trained by evolution
strategies. The connectome itself is never updated during training. The
first task is a toy "addiction" model: a simulated fly that can become
hooked on nicotine (with tolerance and withdrawal) and on variable-ratio
"reels" (a slot-machine-like scrolling stimulus), competing against ordinary
feeding.

## Status: WORK IN PROGRESS

FastBrain (the connectome simulator) and the addiction environment are
tested and working. The policy/trainer interface (`flyrl/policy.py`,
`flyrl/train_es.py`) is being reworked after a diagnostic showed that (1)
photoreceptor input reaches no descending neurons in this connectome model,
and (2) ORN (olfactory receptor neuron) input is not lateralised — both of
which undermine the bilateral-sensing assumptions the policy needs. There is
no trained addicted fly yet.

## Components

| Component | Description |
|---|---|
| `flyrl/fastbrain.py` | Event-driven reimplementation of the reference LIF model (`eon-fly-brain/code/run_pytorch.py`). Bit-exact vs. the reference at dt=0.1 ms on a 1000-step sugar-GRN test. Measured on an Apple M5 Pro CPU: reference 264 s wall-clock per simulated second; FastBrain 2.2 s (batch 32, dt 0.1 ms) and 0.45 s (batch 32, dt 0.5 ms) per simulated second per brain. |
| `flyrl/addiction_env.py` | Gymnasium environment, 12 observation channels (bilateral odor/light + interoceptive signals). Scripted-baseline mean return over 20 seeds: Forager +17.8, Smoker -83.7, Reels -5.7, Greedy -37.4. |
| `flyrl/scripted.py` | Hand-written klinotaxis policies used to sanity-check the environment's reward tuning and produce the baseline table above. |
| `fly3d/`, `view3d.py` | 3D visualisation using the `flybody` MuJoCo fly model: replays an episode in an arena with a sugar droplet, a lit cigarette (smoke wisps, ember) and a phone playing real video as a live texture. The body animation is kinematic/procedural (tripod gait, proboscis, foreleg swipe), not physics- or brain-driven. Currently driven by the scripted policies; see `renders/` for stills and videos. The phone clips are free-licence Mixkit stock footage and are not redistributed here: fetch them with the URLs in `renders/assets/reels/SOURCES.md` (or drop any `*.mp4` into that folder). Render with `MUJOCO_GL=glfw .venv-body/bin/python -m fly3d.scene --mode all --policy greedy --seed 0`. |
| `flyrl/policy.py`, `flyrl/train_es.py` | Policy network and ES trainer that will drive the connectome model on the addiction env. Being reworked (see Status). |

## Design idea

Training a policy to maximize *true* environment return on the addiction
env just produces a forager — nothing interesting happens. To get a fly
that actually behaves "addicted," the plan is to train on a *hijacked*
learning signal instead: a non-habituating nicotine dopamine bonus, a
jackpot bonus for the reels, and a myopic (short-horizon) discount factor,
following the reward-hijacking framing in Redish (2004)'s computational
model of addiction. That "addicted" policy is then compared, on *true*
welfare (real environment return, not the hijacked signal it was trained
on), against a "sober" control policy trained normally on true reward, with
both branching from a shared taxis-pretrained checkpoint so the comparison
isn't confounded by pretraining differences.

## Quick start

```bash
./setup.sh
pytest tests/test_fastbrain.py tests/test_addiction_env.py
python bench_fastbrain.py
python -m flyrl.scripted
```

`setup.sh` clones the four upstream repositories this project depends on at
pinned commits and creates two virtualenvs (`.venv` for the connectome sim
and RL code, `.venv-body` for the MuJoCo body renderer). See its output for
next steps (downloading FlyWire annotations, building the weight cache).

## Caveats

This is a toy model, not a claim about real fly addiction or real
neuroscience. In particular: the neurons are simple LIF units with no
synaptic plasticity and no neuromodulation beyond the hand-designed reward
bonuses described above; the connectome's synaptic weights are frozen
throughout training; and the entire reward structure (dopamine bonus,
jackpot bonus, myopic discount) is hand-designed to produce addiction-like
behavior, not derived from data. Nothing here should be read as evidence
about how nicotine or variable-ratio reinforcement affects real flies or
their brains.

## License and acknowledgements

This repository is licensed under GPL-2.0 (see `LICENSE`), because
`flyrl/fastbrain.py` is a derivative reimplementation of GPL-2.0-licensed
code. See `NOTICE.md` for the full list of upstream projects, datasets, and
papers this work builds on, their licenses, and how each is used.
