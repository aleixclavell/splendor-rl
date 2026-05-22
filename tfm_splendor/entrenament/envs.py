import copy

import gymnasium as gym
import splendor.splendor.gym  # noqa: F401 - registra l'entorn splendor-v1
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from tfm_splendor.entrenament.custom_wrappers import (
    WrapperInfoMascaraAccions,
    WrapperObsAmbMascaraSeguent,
    WrapperRecompensaEntrenament,
    WrapperResultatAvaluacio,
    mask_fn,
)
from gymnasium.wrappers import FlattenObservation, TimeLimit

DEFAULT_WIN_BONUS = 10.0
DEFAULT_LOSS_PENALTY = -10.0
DEFAULT_STEP_PENALTY = 0.05
DEFAULT_EVAL_WIN_REWARD = 1.0
DEFAULT_EVAL_LOSS_REWARD = 0.0
MAX_EPISODE_STEPS = 500

def make_env_ppo(opponents, monitor_file=None, for_training=True):
    env = gym.make("splendor-v1", agents=opponents)
    if for_training:
        env = WrapperRecompensaEntrenament(
            env,
            win_bonus=DEFAULT_WIN_BONUS,
            loss_penalty=DEFAULT_LOSS_PENALTY,
            step_penalty=DEFAULT_STEP_PENALTY,
        )
    else:
        env = WrapperResultatAvaluacio(
            env,
            win_reward=DEFAULT_EVAL_WIN_REWARD,
            loss_reward=DEFAULT_EVAL_LOSS_REWARD,
        )
    env = Monitor(env, filename=monitor_file)
    env = ActionMasker(env, mask_fn)
    return env


def make_env_dsac(opponents, monitor_file=None, flatten_obs=True, for_training=True, eval_loss_reward=0.0):
    env = gym.make("splendor-v1", agents=copy.deepcopy(opponents))
    if for_training:
        env = WrapperRecompensaEntrenament(
            env,
            win_bonus=DEFAULT_WIN_BONUS,
            loss_penalty=DEFAULT_LOSS_PENALTY,
            step_penalty=DEFAULT_STEP_PENALTY,
        )
    else:
        env = WrapperResultatAvaluacio(
            env,
            win_reward=DEFAULT_EVAL_WIN_REWARD,
            loss_reward=eval_loss_reward,
            train_win_bonus=DEFAULT_WIN_BONUS,
            train_loss_penalty=DEFAULT_LOSS_PENALTY,
            train_step_penalty=DEFAULT_STEP_PENALTY,
        )
    env = WrapperObsAmbMascaraSeguent(env)
    if flatten_obs:
        env = FlattenObservation(env)
    env = Monitor(env, filename=monitor_file)
    env = WrapperInfoMascaraAccions(env)
    return env