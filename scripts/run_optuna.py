"""
Llança els estudis Optuna de PPO i DSAC en seqüència.
Pensat per executar al servidor núvol.

Ús:
  python run_optuna.py --mode ppo
  python run_optuna.py --mode dsac
  python run_optuna.py --mode both
"""

import argparse
import os

from splendor.agents.generic.random import RandomAgent
from tfm_splendor.agents.H1Agent import H1Agent


def run_ppo(n_trials: int, timesteps: int):
    from tfm_splendor.entrenament.tuning.ppo import optuna_study
    print(f"\n=== Optuna PPO: {n_trials} trials × {timesteps} steps ===\n")
    optuna_study(
        n_trials=n_trials,
        total_timesteps_trial=timesteps,
        db_path="sqlite:///optuna_ppo.db",
        study_name="optuna_splendor_ppo",
        results_csv="optuna_ppo_results.csv",
        opponents=[RandomAgent(1)],
    )


def run_dsac(n_trials: int, timesteps: int):
    from tfm_splendor.entrenament.tuning.dsac import optuna_study
    print(f"\n=== Optuna DSAC: {n_trials} trials × {timesteps} steps ===\n")
    optuna_study(
        n_trials=n_trials,
        total_timesteps_trial=timesteps,
        n_eval_episodes=20,
        db_path="sqlite:///optuna_dsac.db",
        study_name="optuna_splendor_dsac",
        opponents=[H1Agent(0)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ppo", "dsac", "both"], default="both")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--timesteps-ppo", type=int, default=100_000)
    parser.add_argument("--timesteps-dsac", type=int, default=100_000)
    args = parser.parse_args()

    os.makedirs("dsac_optuna_tmp", exist_ok=True)

    if args.mode in ("ppo", "both"):
        run_ppo(args.trials, args.timesteps_ppo)

    if args.mode in ("dsac", "both"):
        run_dsac(args.trials, args.timesteps_dsac)
