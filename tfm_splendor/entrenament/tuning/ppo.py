from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from sb3_contrib import MaskablePPO
from splendor.agents.generic.random import RandomAgent

from tfm_splendor.entrenament.envs import make_env_ppo


def llista_oponents() -> list[Any]:
    return [RandomAgent(1)]


def optuna_study(
    n_trials: int = 15,
    total_timesteps_trial: int = 100_000,
    db_path: str = "sqlite:///optuna_splendor.db",
    study_name: str = "optuna_splendor_ppo",
    train_monitor_file: str = "optuna_train_monitor",
    results_csv: str = "optuna_results.csv",
    opponents: list[Any] | None = None,
):
    if opponents is None:
        opponents = llista_oponents()

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 20]),
            "gamma": trial.suggest_float("gamma", 0.97, 0.999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.98),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
            "ent_coef": trial.suggest_float("ent_coef", 1e-6, 2e-2, log=True),
            "vf_coef": trial.suggest_float("vf_coef", 0.3, 1.0),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
        }

        train_env = make_env_ppo(
            opponents=opponents,
            monitor_file=train_monitor_file,
            for_training=True,
        )

        try:
            model = MaskablePPO("MlpPolicy", train_env, **params)
            model.learn(total_timesteps=total_timesteps_trial)

            ep_info_buffer = getattr(model, "ep_info_buffer", None)
            if ep_info_buffer:
                rewards = [ep["r"] for ep in ep_info_buffer if "r" in ep]
                return float(np.mean(rewards)) if rewards else 0.0
            return 0.0
        except Exception as exc:
            print(f"[Optuna PPO] Error durant el trial {trial.number}: {exc}")
            return 0.0
        finally:
            train_env.close()

    study = optuna.create_study(
        direction="maximize",
        storage=db_path,
        load_if_exists=True,
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    df = study.trials_dataframe()
    df.to_csv(results_csv, index=False)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study
