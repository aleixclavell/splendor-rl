from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import optuna
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

from tfm_splendor.agents.H1Agent import H1Agent
from tfm_splendor.entrenament.config import dsac_hiperparametres_per_defecte
from tfm_splendor.entrenament.dsac.model import DSACModel
from tfm_splendor.entrenament.dsac.policy import MaskedDiscreteSACPolicy, apply_action_mask
from tfm_splendor.entrenament.envs import make_env_dsac
from tfm_splendor.entrenament.evaluation_utils import avaluar_model


def set_seed(seed: int | None) -> None:
    import random

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def llista_oponents() -> list[Any]:
    return [H1Agent(0)]


def _build_dsac_model(hp: dict, opponents: list[Any]) -> DSACModel:
    env = make_env_dsac(opponents, monitor_file=None, flatten_obs=True, for_training=True)
    assert isinstance(env.action_space, gym.spaces.Discrete), "DiscreteSAC necessita action_space Discrete"
    assert env.observation_space.shape is not None, "Cal una observació amb shape definit"

    state_shape = env.observation_space.shape
    action_shape = env.action_space.n
    action_space = env.action_space
    observation_space = env.observation_space
    env.close()

    device = hp["device"]

    actor_preprocess = Net(state_shape=state_shape, hidden_sizes=hp["hidden_sizes"])
    actor = Actor(
        preprocess_net=actor_preprocess,
        action_shape=action_shape,
        hidden_sizes=(),
        softmax_output=False,
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=hp["actor_lr"])

    critic1_preprocess = Net(state_shape=state_shape, hidden_sizes=hp["hidden_sizes"])
    critic1 = Critic(
        preprocess_net=critic1_preprocess,
        hidden_sizes=(),
        last_size=action_shape,
    ).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=hp["critic_lr"])

    critic2_preprocess = Net(state_shape=state_shape, hidden_sizes=hp["hidden_sizes"])
    critic2 = Critic(
        preprocess_net=critic2_preprocess,
        hidden_sizes=(),
        last_size=action_shape,
    ).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=hp["critic_lr"])

    alpha: float | tuple[float, torch.Tensor, torch.optim.Optimizer]
    alpha = hp["alpha"]
    if hp.get("auto_alpha", True):
        avg_valid = hp.get("avg_valid_actions", 15)
        target_entropy = hp.get("target_entropy_coef", 0.3) * np.log(avg_valid)
        log_alpha = torch.tensor([np.log(hp["alpha"])], requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=hp["alpha_lr"])
        alpha = (target_entropy, log_alpha, alpha_optim)

    policy = MaskedDiscreteSACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        action_space=action_space,
        tau=hp["tau"],
        gamma=hp["gamma"],
        alpha=alpha,
        estimation_step=hp["estimation_step"],
        observation_space=observation_space,
    )
    return DSACModel(policy=policy, hp=hp, apply_action_mask_fn=apply_action_mask)


def optuna_study(
    n_trials: int = 15,
    total_timesteps_trial: int = 100_000,
    n_eval_episodes: int = 20,
    db_path: str = "sqlite:///optuna_dsac_v2.db",
    study_name: str = "optuna_splendor_dsac_v2",
    opponents: list[Any] | None = None,
):
    if opponents is None:
        opponents = llista_oponents()

    from optuna.pruners import MedianPruner

    class _PruneSignal(Exception):
        pass

    def objective(trial: "optuna.Trial") -> float:
        hp = dsac_hiperparametres_per_defecte()
        hp.update({
            # Paràmetres a tunar (rang refinat respecte v1)
            "actor_lr":            trial.suggest_float("actor_lr",            5e-5, 5e-4, log=True),
            "critic_lr":           trial.suggest_float("critic_lr",           5e-5, 1e-3, log=True),
            "tau":                 trial.suggest_float("tau",                 5e-3, 3e-2, log=True),
            "target_entropy_coef": trial.suggest_float("target_entropy_coef", 0.5,  0.95),
            "warmup_steps":        trial.suggest_categorical("warmup_steps",  [1_000, 2_000, 5_000]),
            # Fixats als millors valors coneguts de la v1
            "alpha_lr":            2e-5,
            "gamma":               0.9856,
            "hidden_sizes":        [128, 128],
            "batch_size":          128,
            "estimation_step":     5,
            "update_per_step":     0.1,
            "episode_per_collect": 4,
            "step_per_collect":    None,
            "step_per_epoch":      5_000,
            "training_num":        2,
            "test_num":            2,
            "save_dir":            "./dsac_optuna_tmp",
        })

        set_seed(hp["seed"])
        model = _build_dsac_model(hp, opponents)

        train_envs = DummyVectorEnv([
            lambda: make_env_dsac(opponents, monitor_file=None, flatten_obs=True, for_training=True)
            for _ in range(hp["training_num"])
        ])
        test_envs = DummyVectorEnv([
            lambda: make_env_dsac(opponents, monitor_file=None, flatten_obs=True, for_training=False, eval_loss_reward=0.0)
            for _ in range(hp["test_num"])
        ])
        eval_env = make_env_dsac(
            opponents, monitor_file=None, flatten_obs=True,
            for_training=False, eval_loss_reward=0.0,
        )

        try:
            train_collector = Collector(
                model.policy,
                train_envs,
                VectorReplayBuffer(hp["buffer_size"], hp["training_num"]),
            )
            test_collector = Collector(model.policy, test_envs)

            env_num = hp["training_num"]
            effective_warmup = max(env_num, int(np.ceil(hp["warmup_steps"] / env_num) * env_num))
            train_collector.collect(n_step=effective_warmup, reset_before_collect=True)

            max_epoch = max(1, int(np.ceil(total_timesteps_trial / hp["step_per_epoch"])))
            best_avg_reward = -np.inf
            report_step = 0

            def test_fn(epoch: int, _env_step: int) -> None:
                nonlocal best_avg_reward, report_step
                metrics = avaluar_model(
                    model,
                    n_episodes=n_eval_episodes,
                    env=eval_env,
                    print_summary=False,
                    mode="dsac",
                )
                avg_reward = float(metrics["avg_reward"])
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                trial.report(avg_reward, report_step)
                report_step += 1
                if trial.should_prune():
                    raise _PruneSignal()

            trainer = OffpolicyTrainer(
                policy=model.policy,
                train_collector=train_collector,
                test_collector=test_collector,
                max_epoch=max_epoch,
                step_per_epoch=hp["step_per_epoch"],
                step_per_collect=hp["step_per_collect"],
                episode_per_collect=hp["episode_per_collect"],
                episode_per_test=hp["episode_per_test"],
                batch_size=hp["batch_size"],
                update_per_step=hp["update_per_step"],
                test_fn=test_fn,
                test_in_train=False,
                verbose=False,
                show_progress=False,
            )

            try:
                trainer.run()
            except _PruneSignal:
                raise optuna.exceptions.TrialPruned()
            except Exception as exc:
                print(f"[Trial {trial.number}] Error durant entrenament: {exc}")
                return 0.0

            return best_avg_reward
        finally:
            train_envs.close()
            test_envs.close()
            eval_env.close()

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(
        direction="maximize",
        storage=db_path,
        load_if_exists=True,
        study_name=study_name,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    df = study.trials_dataframe()
    df.to_csv("optuna_dsac_results.csv", index=False)

    print("\n=== Millor trial ===")
    best = study.best_trial
    print(f"  Avg reward: {best.value:.4f}")
    print("  Hiperparàmetres:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    return study
