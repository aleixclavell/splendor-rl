from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import copy

from tfm_splendor.entrenament.envs import make_env_ppo
from tfm_splendor.entrenament.evaluation_utils import avaluar_model
from tfm_splendor.entrenament.utils import read_codecarbon_csv


class CallbackEntrenament(BaseCallback):
    """
    Registra mètriques de l'entrenament al progress.csv i TensorBoard
    al final de cada rollout (via _on_rollout_end, alineat amb n_steps).

    Mètriques registrades:
        Rollout (ep_rew_mean i ep_len_mean les registra SB3 automàticament):
            rollout/ep_len_std, ep_rew_std, ep_len_median, ep_rew_median, ep_count

        Energia (CodeCarbon, memòria + CSV opcional):
            codecarbon/energy_kwh_delta          -> energia d'aquest bloc de steps
            codecarbon/energy_kwh_total          -> energia acumulada
            codecarbon/energy_kwh_cpu/gpu/ram    -> desglossat per component
            codecarbon/emissions_kg_delta        -> emissions CO2 d'aquest bloc (CSV)
            codecarbon/emissions_kg_total        -> emissions CO2 acumulades (CSV)
            codecarbon/power_w_cpu/gpu/ram       -> potència instantània (CSV)
            codecarbon/emissions_rate            -> taxa d'emissions (CSV)

        Eficiència energètica:
            codecarbon/energy_kwh_per_1k_steps   -> kWh per 1000 passos
            codecarbon/energy_kwh_per_mean_reward -> kWh per unitat de recompensa mitjana

        PPO intern (SB3 automàtic):
            train/entropy_loss, train/value_loss, train/policy_gradient_loss,
            train/approx_kl, train/clip_fraction
    """

    def __init__(
        self,
        tracker,
        train_monitor_file="train_monitor",
        verbose=0,
        emissions_csv=None,
    ):
        super().__init__(verbose)
        self.tracker = tracker
        self.train_monitor_file = train_monitor_file
        self.emissions_csv = emissions_csv

    def _read_emissions(self):
        """
        Llegeix energia del tracker en memòria (precís) i emissions/potència del CSV.
        Retorna un dict amb totes les mètriques unificades.
        """
        result = {
            "total_energy": np.nan,
            "cpu_energy": np.nan, "gpu_energy": np.nan, "ram_energy": np.nan,
            "total_emissions": np.nan,
            "cpu_power": np.nan, "gpu_power": np.nan, "ram_power": np.nan,
            "emissions_rate": np.nan,
        }
        try:
            result["total_energy"] = float(self.tracker._total_energy.kWh)
            result["cpu_energy"] = float(self.tracker._total_cpu_energy.kWh)
            result["gpu_energy"] = float(self.tracker._total_gpu_energy.kWh)
            result["ram_energy"] = float(self.tracker._total_ram_energy.kWh)
        except Exception:
            pass

        if self.emissions_csv:
            try:
                self.tracker.flush()
            except Exception:
                pass
            csv_data = read_codecarbon_csv(self.emissions_csv)
            total_emissions = csv_data.get("emissions")
            if total_emissions is not None:
                result["total_emissions"] = total_emissions
            for key in ("cpu_power", "gpu_power", "ram_power", "emissions_rate"):
                if key in csv_data:
                    result[key] = csv_data[key]

        return result

    def _read_monitor_stats(self):
        ep_info_buffer = getattr(self.model, "ep_info_buffer", None)
        if not ep_info_buffer:
            return {
                "count": 0, "len_mean": np.nan, "rew_mean": np.nan,
                "len_std": np.nan, "rew_std": np.nan,
                "len_median": np.nan, "rew_median": np.nan,
            }
        lengths = [ep.get("l", np.nan) for ep in ep_info_buffer]
        rewards = [ep.get("r", np.nan) for ep in ep_info_buffer]
        lengths = [x for x in lengths if np.isfinite(x)]
        rewards = [x for x in rewards if np.isfinite(x)]
        return {
            "count": len(lengths),
            "len_mean": float(np.mean(lengths)) if lengths else np.nan,
            "rew_mean": float(np.mean(rewards)) if rewards else np.nan,
            "len_std": float(np.std(lengths)) if lengths else np.nan,
            "rew_std": float(np.std(rewards)) if rewards else np.nan,
            "len_median": float(np.median(lengths)) if lengths else np.nan,
            "rew_median": float(np.median(rewards)) if rewards else np.nan,
        }

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        stats = self._read_monitor_stats()
        self.logger.record("rollout/ep_len_std",    stats["len_std"])
        self.logger.record("rollout/ep_rew_std",    stats["rew_std"])
        self.logger.record("rollout/ep_len_median", stats["len_median"])
        self.logger.record("rollout/ep_rew_median", stats["rew_median"])
        self.logger.record("rollout/ep_count",      stats["count"])

        em = self._read_emissions()
        self.logger.record("codecarbon/energy_kwh_total",   em["total_energy"])
        self.logger.record("codecarbon/energy_kwh_cpu",     em["cpu_energy"])
        self.logger.record("codecarbon/energy_kwh_gpu",     em["gpu_energy"])
        self.logger.record("codecarbon/energy_kwh_ram",     em["ram_energy"])
        self.logger.record("codecarbon/emissions_kg_total", em["total_emissions"])
        self.logger.record("codecarbon/power_w_cpu",        em["cpu_power"])
        self.logger.record("codecarbon/power_w_gpu",        em["gpu_power"])
        self.logger.record("codecarbon/power_w_ram",        em["ram_power"])
        self.logger.record("codecarbon/emissions_rate",     em["emissions_rate"])

        if self.verbose:
            print(
                f"[TRAIN {self.num_timesteps} steps] "
                f"mean_reward={stats['rew_mean']:.3f}  "
                f"total_energy={em['total_energy']:.6f} kWh"
            )


class CallbackAvaluacioPeriodica(BaseCallback):
    def __init__(
        self,
        opponent_factory,
        eval_freq=2_048 * 5,
        n_eval_episodes=200,
        monitor_file="eval_monitor_periodic.csv",
        verbose=1,
        tracker=None,
    ):
        super().__init__(verbose)
        self.opponent_factory = opponent_factory
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.monitor_file = monitor_file
        self.eval_env = None
        self.tracker = tracker

    def _init_callback(self) -> None:
        eval_opponents = [copy.deepcopy(op) for op in self.opponent_factory()]
        self.eval_env = make_env_ppo(
            opponents=eval_opponents,
            monitor_file=self.monitor_file,
            for_training=False,
        )

    def _eval_and_log(self, step: int) -> None:
        metrics = avaluar_model(
            model=self.model,
            n_episodes=self.n_eval_episodes,
            env=self.eval_env,
            print_summary=False,
        )
        self.logger.record("eval/winrate",           float(metrics["winrate"]))
        self.logger.record("eval/avg_score",         float(metrics["avg_score"]))
        self.logger.record("eval/mean_nobles",       metrics["avg_nobles"])
        self.logger.record("eval/mean_bought_cards", metrics["avg_cartes"])

        if self.tracker is not None:
            try:
                self.logger.record("eval/energy_kwh_total", float(self.tracker._total_energy.kWh))
            except Exception:
                pass

        self.logger.dump(step)
        if self.verbose:
            print(
                f"[EVAL {step} steps] "
                f"winrate={metrics['winrate']:.2%}  "
                f"avg_score={metrics['avg_score']:.2f}  "
                f"avg_reward={metrics['avg_reward']:.3f}  "
                f"nobles={metrics['avg_nobles']:.2f}  "
                f"cards={metrics['avg_cartes']:.2f}  "
                f"episodes={self.n_eval_episodes}  "
            )

    def _on_training_start(self) -> None:
        self._eval_and_log(0)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
        self._eval_and_log(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        if self.num_timesteps % self.eval_freq != 0:
            self._eval_and_log(self.num_timesteps)
        if self.eval_env is not None:
            self.eval_env.close()
