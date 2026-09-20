"""Task C: innate-approach ("taxis") pretraining support for
`flyrl.train_es --mode taxis`.

Rationale: before ES has to jointly learn steering AND cope with the
addiction dynamics (hijacked/welfare modes), it helps to first pretrain the
encoder/decoder to do the much simpler thing the connectome's sensory
entry points are actually anatomically wired for -- walk toward whatever
single modality (food odor, smoke odor, or reels light) is currently
visible. This module implements that as a THIN WRAPPER around
FlyAddictionEnv/VecFlyAddictionEnv (imported, never modified): it reads
env attributes (`.pos`, `.sources`, `.params.r_consume`) and masks the obs
array returned to the policy; it does not touch flyrl/addiction_env.py.

Per generation, one population evaluation = 3 sub-episodes of
`n_substeps` steps (spec: 120), one per modality in MODALITIES. In each
sub-episode, `mask_obs` zeroes every obs channel that isn't the target
modality's own (L/R sensory pair + its taste/jackpot channel); the three
interoceptive channels (hunger, nicotine, withdrawal) are always left
visible. Fitness for that sub-episode is DENSE:

    (initial_distance_to_target - final_distance_to_target)
    + 0.01 * (steps spent within r_consume of the target)

averaged over the 3 sub-episodes to give one fitness value per population
member. Common random numbers (every population member sees the SAME
environment realization for a given modality within a generation) and
antithetic sampling are preserved: this module reuses train_es.py's own
per-sub-env common-seed reset pattern, and antithetic pairing is entirely
handled by train_es.build_population/es_gradient upstream of this module
(run_taxis_generation just consumes whatever theta_pop it's given).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flyrl.addiction_env import FlyAddictionEnv, VecFlyAddictionEnv

MODALITIES = ("food", "smoke", "reels")

_CHANNEL_INDEX = {name: i for i, name in enumerate(FlyAddictionEnv.obs_channels)}
_INTEROCEPTIVE = [_CHANNEL_INDEX["hunger"], _CHANNEL_INDEX["nicotine"], _CHANNEL_INDEX["withdrawal"]]

# obs channel indices to KEEP per modality sub-episode (all others zeroed).
MODALITY_KEEP_CHANNELS = {
    "food": [_CHANNEL_INDEX["food_odor_L"], _CHANNEL_INDEX["food_odor_R"],
             _CHANNEL_INDEX["sugar_taste"]] + _INTEROCEPTIVE,
    "smoke": [_CHANNEL_INDEX["smoke_odor_L"], _CHANNEL_INDEX["smoke_odor_R"],
              _CHANNEL_INDEX["nicotine_taste"]] + _INTEROCEPTIVE,
    "reels": [_CHANNEL_INDEX["reels_light_L"], _CHANNEL_INDEX["reels_light_R"],
              _CHANNEL_INDEX["reels_jackpot"]] + _INTEROCEPTIVE,
}

DIST_REDUCTION_STEP_BONUS = 0.01  # per-step bonus while within r_consume of the target


def mask_obs(obs: np.ndarray, modality: str) -> np.ndarray:
    """Zero every obs channel not belonging to `modality` (see
    MODALITY_KEEP_CHANNELS). obs: (..., 12) -> (..., 12), same dtype."""
    keep = MODALITY_KEEP_CHANNELS[modality]
    masked = np.zeros_like(obs)
    masked[..., keep] = obs[..., keep]
    return masked


def _vec_reset_common_seed(vec_env: VecFlyAddictionEnv, seed: int) -> np.ndarray:
    """Reset every sub-env of `vec_env` with the IDENTICAL seed (common
    random numbers across the ES population), mirroring
    train_es._vec_reset_common_seed exactly (duplicated here rather than
    imported to keep this module import-independent of train_es)."""
    obs_list = []
    for env in vec_env.envs:
        obs, _info = env.reset(seed=seed)
        obs_list.append(obs)
    vec_env._last_obs = obs_list
    vec_env._done[:] = False
    return np.stack(obs_list, axis=0)


def _distances(vec_env: VecFlyAddictionEnv, modality: str) -> np.ndarray:
    return np.array([float(np.linalg.norm(env.pos - env.sources[modality]))
                      for env in vec_env.envs], dtype=np.float64)


@dataclass
class TaxisResult:
    fitness: np.ndarray                  # (B,) averaged dense fitness over the 3 sub-episodes
    reach_rate: dict                     # modality -> (B,) bool, reached r_consume at least once
    dist_reduction: dict                 # modality -> (B,) float, initial_dist - final_dist
    mean_steps_within: dict              # modality -> (B,) int, steps spent within r_consume


def run_taxis_generation(policy, vec_env: VecFlyAddictionEnv, theta_pop: np.ndarray,
                          seed: int, n_substeps: int = 120) -> TaxisResult:
    """Evaluate `theta_pop` (B, N_PARAMS) as one taxis-pretraining
    generation: 3 sub-episodes of `n_substeps` steps (food, smoke, reels),
    common random numbers within each modality across the population."""
    B = theta_pop.shape[0]
    assert vec_env.num_envs == B and policy.batch == B
    policy.set_params(theta_pop)

    fitness_sum = np.zeros(B, dtype=np.float64)
    reach_rate = {}
    dist_reduction = {}
    mean_steps_within = {}

    for mi, modality in enumerate(MODALITIES):
        policy.reset()
        # distinct per-modality seed, but IDENTICAL across the whole
        # population for this generation (common random numbers).
        mod_seed = seed + mi * 1_000_003
        obs = _vec_reset_common_seed(vec_env, mod_seed)

        init_dist = _distances(vec_env, modality)
        r_consume = vec_env.envs[0].params.r_consume
        steps_within = np.zeros(B, dtype=np.int64)
        reached = np.zeros(B, dtype=bool)

        obs = mask_obs(obs, modality)
        for _ in range(n_substeps):
            actions = policy.act(obs)
            obs, _rewards, _dones, _infos = vec_env.step(actions)
            obs = mask_obs(obs, modality)
            dist = _distances(vec_env, modality)
            hit = dist <= r_consume
            steps_within += hit.astype(np.int64)
            reached |= hit

        final_dist = _distances(vec_env, modality)
        reduction = init_dist - final_dist
        sub_fitness = reduction + DIST_REDUCTION_STEP_BONUS * steps_within

        fitness_sum += sub_fitness
        reach_rate[modality] = reached
        dist_reduction[modality] = reduction
        mean_steps_within[modality] = steps_within

    fitness = fitness_sum / len(MODALITIES)
    return TaxisResult(fitness=fitness, reach_rate=reach_rate,
                        dist_reduction=dist_reduction, mean_steps_within=mean_steps_within)
