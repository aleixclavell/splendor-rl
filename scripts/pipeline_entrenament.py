"""
Executa una seqüència d'etapes d'entrenament (PPO o DSAC) de manera encadenada.

Cada etapa pot:
  - Partir del model guardat a l'etapa anterior (PPO).
  - Canviar d'oponents (Random → H1Agent → selfplay, etc.).
  - Usar mode "dsac" (DSAC sempre crea model nou; no suporta continuació de pesos).

Ús bàsic:
  Definir les etapes al bloc __main__ i cridar executar_pipeline().

Nota sobre mode mixt: no té sentit barrejar PPO i DSAC en la mateixa pipeline
perquè les arquitectures i formats de fitxer (.zip vs .pth) són incompatibles.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional
import os

from codecarbon import EmissionsTracker
from sb3_contrib import MaskablePPO
from splendor.agents.generic.random import RandomAgent
from stable_baselines3.common.logger import configure

from tfm_splendor.agents import H2Agent
from tfm_splendor.agents.H1Agent import H1Agent
from tfm_splendor.agents.TrainedPPOAgent import TrainedPPOAgent
from tfm_splendor.agents.TrainedDSACAgent import TrainedDSACAgent
from tfm_splendor.entrenament.config import ppo_hiperparametres_finals, ppo_hiperparametres_h1, ppo_hiperparametres_per_defecte, ppo_hiperparametres_optuna, dsac_hiperparametres_actuals, dsac_hiperparametres_per_defecte, ppo_hiperparametres_optuna_v2
from tfm_splendor.entrenament.envs import make_env_ppo
from tfm_splendor.entrenament.callbacks import CallbackAvaluacioPeriodica, CallbackEntrenament
from tfm_splendor.entrenament.evaluation_utils import avaluar_model
from tfm_splendor.entrenament.utils import read_codecarbon_csv
from tfm_splendor.entrenament.artifact_utils import (
    enviar_mail,
    registrar_hiperparametres,
    preparar_execucio,
)
from dsac_splendor import entrenar_dsac, carregar_model_dsac, reset_critic_dsac
from ppo_splendor import reset_critic_ppo

LOG_GRANULARITY = 2_048
N_EVAL_EPISODES_PERIODIC = 100
N_EVAL_EPISODES_FINAL = 200


@dataclass
class Etapa:
    nom: str
    total_timesteps: int
    opponents_fn: Callable[[Optional[str]], List]
    # opponents_fn(prev_model_path) → llista d'oponents
    # Per selfplay PPO: lambda prev: [TrainedPPOAgent(1, MaskablePPO.load(prev))]
    mode: str = "ppo"
    # "ppo": carrega i continua el model anterior (.zip)
    # "dsac": sempre crea model nou (l'API actual no suporta continuació)
    hp_fn: Optional[Callable[[], dict]] = None
    eval_opponents_fn: Optional[Callable[[Optional[str]], List]] = None
    # eval_opponents_fn: oponents per a l'avaluació periòdica (DSAC).
    # Si és None, s'usen els mateixos que opponents_fn.
    reset_critic: bool = False
    # reset_critic=True: reinicialitza els pesos del critic (value net) en carregar
    # el model de l'etapa anterior. Útil per a curriculum learning quan el critic
    # ha sobreajustat a l'oponent anterior. L'actor conserva els pesos apresos.


# ---------------------------------------------------------------------------
# Etapes individuals
# ---------------------------------------------------------------------------

def _executar_etapa_ppo(
    etapa: Etapa,
    idx: int,
    experiment_nom: str,
    prev_model_path: Optional[str],
):
    run_name = f"{experiment_nom}/e{idx:02d}p_{etapa.nom}"
    paths = preparar_execucio(run_name)

    train_monitor_file = os.path.join(paths["monitor_dir"], "train_monitor")
    eval_monitor_file = os.path.join(paths["monitor_dir"], "eval_monitor_periodic")
    nom_model = os.path.join(paths["model_dir"], f"model_e{idx:02d}p_{etapa.nom}")
    emissions_file = os.path.join(paths["emissions_dir"], "emissions.csv")

    opponents = etapa.opponents_fn(prev_model_path)
    train_env = make_env_ppo(opponents, monitor_file=train_monitor_file, for_training=True)

    if prev_model_path and not prev_model_path.endswith(".pth"):
        _OVERRIDABLE_HP = {"learning_rate", "ent_coef", "clip_range", "gamma", "gae_lambda",
                           "n_steps", "batch_size", "n_epochs", "vf_coef", "max_grad_norm"}
        custom_hp = {k: v for k, v in (etapa.hp_fn or ppo_hiperparametres_per_defecte)().items()
                     if k in _OVERRIDABLE_HP}
        model = MaskablePPO.load(prev_model_path.removesuffix(".zip"), env=train_env,
                                 custom_objects=custom_hp or None)
        if etapa.reset_critic:
            reset_critic_ppo(model)
    else:
        model = MaskablePPO("MlpPolicy", train_env, **(etapa.hp_fn or ppo_hiperparametres_per_defecte)())

    registrar_hiperparametres(
        model, paths["config_dir"],
        opponents=opponents,
        total_timesteps=etapa.total_timesteps,
        initial_model_path=prev_model_path if prev_model_path and not prev_model_path.endswith(".pth") else None,
    )

    new_logger = configure(paths["logs_dir"], ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    os.environ["CODECARBON_COUNTRY_ISO_CODE"] = "ESP"
    os.environ["CODECARBON_REGION"] = "catalonia"
    tracker = EmissionsTracker(
        project_name=f"Entrenament PPO {etapa.nom}",
        measure_power_secs=10,
        output_file=emissions_file,
        log_level="error",
    )

    cb_train = CallbackEntrenament(
        tracker=tracker,
        train_monitor_file=train_monitor_file,
        verbose=0,
        emissions_csv=emissions_file,
    )
    eval_opponents_fn = etapa.eval_opponents_fn or (lambda _: [H1Agent(1)])
    cb_eval = CallbackAvaluacioPeriodica(
        opponent_factory=lambda: eval_opponents_fn(prev_model_path),
        eval_freq=LOG_GRANULARITY * 5,
        n_eval_episodes=N_EVAL_EPISODES_PERIODIC,
        monitor_file=eval_monitor_file,
        verbose=0,
        tracker=tracker,
    )

    tracker.start()
    emissions = 0.0
    try:
        model.learn(
            total_timesteps=etapa.total_timesteps,
            callback=[cb_train, cb_eval],
            reset_num_timesteps=True,
            log_interval=1,
        )
        model.save(nom_model)
        saved_path = nom_model + ".zip"
    finally:
        emissions = tracker.stop() or 0.0
        print(f"Emissions etapa '{etapa.nom}': {emissions:.6f} kg CO2")
        train_env.close()

    return model, saved_path, emissions


def _executar_etapa_dsac(
    etapa: Etapa,
    idx: int,
    experiment_nom: str,
    prev_model_path: Optional[str],
):
    run_name = f"{experiment_nom}/e{idx:02d}d_{etapa.nom}"
    paths = preparar_execucio(run_name)

    nom_model = os.path.join(paths["model_dir"], f"model_e{idx:02d}d_{etapa.nom}")
    emissions_file = os.path.join(paths["emissions_dir"], "emissions.csv")

    opponents = etapa.opponents_fn(prev_model_path)
    eval_opponents = etapa.eval_opponents_fn(prev_model_path) if etapa.eval_opponents_fn else None

    prev_dsac = None
    if prev_model_path and prev_model_path.endswith(".pth"):
        from dsac_splendor import carregar_model_dsac
        prev_dsac = carregar_model_dsac(prev_model_path, opponents=opponents)
        print(f"  → Continuant des de pesos anteriors: {prev_model_path}")
        if etapa.reset_critic:
            reset_critic_dsac(prev_dsac)

    model, _ = entrenar_dsac(
        opponents=opponents,
        total_timesteps=etapa.total_timesteps,
        model=prev_dsac,
        model_path=nom_model,
        emissions_file=emissions_file,
        train_monitor_dir=os.path.join(paths["monitor_dir"], "train"),
        test_monitor_dir=os.path.join(paths["monitor_dir"], "eval_periodic"),
        logger_dir=paths["logs_dir"],
        hp_fn=(etapa.hp_fn or dsac_hiperparametres_actuals),
        config_dir=paths["config_dir"],
        eval_opponents=eval_opponents,
        initial_model_path=prev_model_path if prev_model_path and prev_model_path.endswith(".pth") else None,
    )

    saved_path = nom_model + ".pth"
    emissions = read_codecarbon_csv(emissions_file).get("emissions", 0.0)
    print(f"Emissions etapa '{etapa.nom}': {emissions:.6f} kg CO2")
    return model, saved_path, emissions


# ---------------------------------------------------------------------------
# Orquestrador principal
# ---------------------------------------------------------------------------

def executar_pipeline(
    etapes: List[Etapa],
    experiment_nom: str = None,
    eval_opponents_fn: Callable[[Optional[str]], List] = None,
    n_eval_episodes: int = N_EVAL_EPISODES_FINAL,
    initial_model_path: Optional[str] = None,
):
    """
    Executa totes les etapes en seqüència i fa una avaluació final.

    Args:
        etapes:           Llista d'Etapa que defineix el pla d'entrenament.
        experiment_nom:   Nom base de l'experiment (es genera automàticament si és None).
        eval_opponents_fn: Factory d'oponents per a l'avaluació final.
                           Rep prev_model_path i retorna llista d'agents.
                           Per defecte: [H1Agent(1)].
        n_eval_episodes:  Episodis de l'avaluació final.
    """
    if experiment_nom is None:
        experiment_nom = "pl_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"PIPELINE: {experiment_nom}  ({len(etapes)} etapes)")
    print(f"{'='*60}\n")

    prev_model_path: Optional[str] = initial_model_path
    current_model = None
    total_emissions = 0.0
    etapes_resum = []

    for i, etapa in enumerate(etapes):
        print(f"\n{'='*60}")
        print(
            f"ETAPA {i+1}/{len(etapes)} [{etapa.mode.upper()}]: "
            f"{etapa.nom}  ({etapa.total_timesteps:,} passos)"
        )
        print(f"{'='*60}")

        if etapa.mode == "dsac":
            current_model, prev_model_path, emissions = _executar_etapa_dsac(
                etapa, i + 1, experiment_nom, prev_model_path
            )
        else:
            current_model, prev_model_path, emissions = _executar_etapa_ppo(
                etapa, i + 1, experiment_nom, prev_model_path
            )

        total_emissions += emissions
        etapes_resum.append({
            "nom": etapa.nom,
            "mode": etapa.mode,
            "timesteps": etapa.total_timesteps,
            "emissions": emissions,
            "model_path": prev_model_path,
        })

    # Avaluació final
    print(f"\n{'='*60}")
    print("AVALUACIÓ FINAL")
    print(f"{'='*60}")

    paths_final = preparar_execucio(f"{experiment_nom}/eval_final")
    final_monitor = os.path.join(paths_final["monitor_dir"], "eval_final")
    emissions_eval_file = os.path.join(paths_final["emissions_dir"], "emissions_eval.csv")

    eval_opponents = eval_opponents_fn(prev_model_path) if eval_opponents_fn else [H1Agent(1)]
    eval_mode = etapes[-1].mode if etapes else "ppo"

    if eval_mode == "dsac":
        final_model = carregar_model_dsac(prev_model_path, opponents=eval_opponents)
    else:
        final_model = MaskablePPO.load(prev_model_path)

    os.environ["CODECARBON_COUNTRY_ISO_CODE"] = "ESP"
    os.environ["CODECARBON_REGION"] = "catalonia"
    tracker_eval = EmissionsTracker(
        project_name=f"Avaluació Final {experiment_nom}",
        measure_power_secs=10,
        output_file=emissions_eval_file,
        log_level="error",
    )
    tracker_eval.start()
    try:
        metrics = avaluar_model(
            final_model,
            opponents=eval_opponents,
            n_episodes=n_eval_episodes,
            monitor_file=final_monitor,
            mode=eval_mode,
        )
    finally:
        emissions_eval = tracker_eval.stop() or 0.0

    total_emissions += emissions_eval

    etapes_txt = "\n".join(
        f"  Etapa {j+1} [{e['mode'].upper()}]: {e['nom']}"
        f" — {e['timesteps']:,} passos"
        f" — {e['emissions']:.6f} kg CO2"
        for j, e in enumerate(etapes_resum)
    )
    enviar_mail(
        subject=f"Pipeline finalitzat — {experiment_nom}",
        body=(
            f"La pipeline d'entrenament ha finalitzat.\n\n"
            f"Experiment: {experiment_nom}\n"
            f"Etapes:\n{etapes_txt}\n\n"
            f"Emissions totals (entrenament + avaluació): {total_emissions:.6f} kg CO2\n\n"
            f"*** Avaluació final ***\n"
            f"Winrate: {metrics['winrate']:.2%}\n"
            f"Recompensa mitjana: {metrics['avg_reward']:.3f}\n"
            f"Nobles (mitjana): {metrics['avg_nobles']:.2f}\n"
            f"Cartes comprades (mitjana): {metrics['avg_cartes']:.2f}\n"
        ),
    )

    return final_model, metrics, etapes_resum


def optuna_study_ppo(
    n_trials: int = 20,
    total_timesteps_trial: int = 200_000,
    opponents_fn: Optional[Callable[[Optional[str]], List]] = None,
    db_path: str = "sqlite:///optuna_pipeline_ppo.db",
    study_name: str = "optuna_pipeline_ppo",
):
    """
    Cerca els hiperparàmetres learning_rate, clip_range i ent_coef amb Optuna.
    La resta de paràmetres prenen els valors de ppo_hiperparametres_per_defecte().

    Mètrica d'optimització: ep_rew_mean dels últims episodis d'entrenament.

    Args:
        n_trials:               Nombre de trials Optuna.
        total_timesteps_trial:  Passos d'entrenament per trial.
        opponents_fn:           Factory d'oponents per entrenament (per defecte H1Agent).
        db_path:                URI de la base de dades Optuna.
        study_name:             Nom de l'estudi Optuna.
    """
    import optuna

    if opponents_fn is None:
        opponents_fn = lambda _: [H1Agent(1)]

    def objective(trial: "optuna.Trial") -> float:
        hp = ppo_hiperparametres_per_defecte()
        hp["learning_rate"] = trial.suggest_float("learning_rate", 5e-5, 2e-4, log=True)
        hp["clip_range"]    = trial.suggest_float("clip_range",    0.08, 0.2)
        hp["ent_coef"]      = trial.suggest_float("ent_coef",      3e-3, 3e-2, log=True)

        run_name = f"optuna_{study_name}/trial_{trial.number:03d}"
        paths = preparar_execucio(run_name)
        train_monitor_file = os.path.join(paths["monitor_dir"], "train_monitor")
        nom_model          = os.path.join(paths["model_dir"],   f"model_trial_{trial.number:03d}")

        opponents  = opponents_fn(None)
        train_env  = make_env_ppo(opponents, monitor_file=train_monitor_file, for_training=True)
        model      = MaskablePPO("MlpPolicy", train_env, **hp)
        new_logger = configure(paths["logs_dir"], ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

        try:
            model.learn(
                total_timesteps=total_timesteps_trial,
                reset_num_timesteps=True,
                log_interval=1,
            )
            model.save(nom_model)
        finally:
            train_env.close()

        buf = getattr(model, "ep_info_buffer", None)
        rewards = [ep["r"] for ep in buf if "r" in ep] if buf else []
        return float(sum(rewards) / len(rewards)) if rewards else 0.0

    study = optuna.create_study(
        direction="maximize",
        storage=db_path,
        load_if_exists=True,
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\nMillor trial #{best.number}  —  avg_reward: {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    return study

def optuna_study_dsac():
    pass


def prova_pipeline_1_ppo():
    pipeline = [
        Etapa(
            nom="vs_random",
            total_timesteps=200_000,
            opponents_fn=lambda _: [RandomAgent(1)],
            mode="ppo",
            hp_fn=ppo_hiperparametres_per_defecte,
            eval_opponents_fn=lambda _: [H1Agent(1)],
        )
    ]
    executar_pipeline(
        etapes=pipeline,
        eval_opponents_fn=lambda _: [H1Agent(1)],
        n_eval_episodes=N_EVAL_EPISODES_FINAL,
    )

# dsac contra random 200k passos
def prova_pipeline_1_dsac():
    pipeline = [
        Etapa(
            nom="vs_random",
            total_timesteps=200_000,
            opponents_fn=lambda _: [RandomAgent(1)],
            mode="dsac",
            hp_fn=dsac_hiperparametres_per_defecte,
            eval_opponents_fn=lambda _: [H1Agent(1)],
        )
    ]
    executar_pipeline(
        etapes=pipeline,
        eval_opponents_fn=lambda _: [H1Agent(1)],
        n_eval_episodes=N_EVAL_EPISODES_FINAL,
    )

# PPO contra H1Agent 200k passos 
def prova_pipeline_2_ppo():
    pipeline = [
        Etapa(
            nom="vs_h1",
            total_timesteps=200_000,
            opponents_fn=lambda _: [H1Agent(1)],
            mode="ppo",
            hp_fn=ppo_hiperparametres_per_defecte,
            eval_opponents_fn=lambda _: [H1Agent(1)],
        )
    ]
    executar_pipeline(
        etapes=pipeline,
        eval_opponents_fn=lambda _: [H1Agent(1)],
        n_eval_episodes=N_EVAL_EPISODES_FINAL,
    )

def prova_pipeline_2_dsac():
    pipeline = [
        Etapa(
            nom="vs_h1",
            total_timesteps=200_000,
            opponents_fn=lambda _: [H1Agent(1)],
            mode="dsac",
            hp_fn=dsac_hiperparametres_per_defecte,
            eval_opponents_fn=lambda _: [H1Agent(1)],
        )
    ]
    executar_pipeline(
        etapes=pipeline,
        eval_opponents_fn=lambda _: [H1Agent(1)],
        n_eval_episodes=N_EVAL_EPISODES_FINAL,
    )

# prova 3. random 100k + h1 100k amb PPO
def prova_pipeline_3_ppo():
    pipeline = [
        Etapa(
            nom="vs_random",
            total_timesteps=100_000,
            opponents_fn=lambda _: [RandomAgent(1)],
            mode="ppo",
            hp_fn=ppo_hiperparametres_per_defecte,
            eval_opponents_fn=lambda _: [H1Agent(1)],
        ),
        Etapa(
            nom="vs_h1",
            total_timesteps=100_000,
            opponents_fn=lambda _: [H1Agent(1)],
            mode="ppo",
            hp_fn=ppo_hiperparametres_per_defecte,
            eval_opponents_fn=lambda _: [H1Agent(1)],
        )
    ]
    executar_pipeline(
        etapes=pipeline,
        eval_opponents_fn=lambda _: [H1Agent(1)],
        n_eval_episodes=N_EVAL_EPISODES_FINAL,
    )

# prova 3. random 100k + h1 100k amb DSAC
def prova_pipeline_3_dsac():
    pipeline = [
        Etapa(
            nom="vs_random",
            total_timesteps=100_000,
            opponents_fn=lambda _: [RandomAgent(1)],
            mode="dsac",
            hp_fn=dsac_hiperparametres_per_defecte,
            eval_opponents_fn=lambda _: [H1Agent(1)],
        ),
        Etapa(
            nom="vs_h1",
            total_timesteps=100_000,
            opponents_fn=lambda _: [H1Agent(1)],
            mode="dsac",
            hp_fn=dsac_hiperparametres_per_defecte,
            eval_opponents_fn=lambda _: [H1Agent(1)],
        )
    ]
    executar_pipeline(
        etapes=pipeline,
        eval_opponents_fn=lambda _: [H1Agent(1)],
        n_eval_episodes=N_EVAL_EPISODES_FINAL,
    )

# prova 4. random 40k + h1 40k + selfplay 4x40k amb PPO
def prova_pipeline_4_ppo():
    n_selfplays = 4
    pipeline = [
        Etapa(
            nom="vs_random",
            total_timesteps=40_906, 
            opponents_fn=lambda _: [RandomAgent(1)],
            mode="ppo",
            hp_fn=ppo_hiperparametres_per_defecte,
            eval_opponents_fn=lambda _: [H1Agent(1)],
        ),
        *[
            Etapa(
                nom=f"selfplay_{i}",
                total_timesteps=40_906, 
                opponents_fn=lambda prev: [TrainedPPOAgent(1, MaskablePPO.load(prev))],
                mode="ppo",
                hp_fn=ppo_hiperparametres_per_defecte,
                eval_opponents_fn=lambda _: [H1Agent(1)],
            )
            for i in range(1, n_selfplays + 1)
        ]
    ]
    executar_pipeline(
        etapes=pipeline,
        eval_opponents_fn=lambda _: [H1Agent(1)],
        n_eval_episodes=N_EVAL_EPISODES_FINAL,
    )

# prova 4. random 40k + h1 40k + selfplay 4x40k amb DSAC
def prova_pipeline_4_dsac():
    n_selfplays = 4
    pipeline = [
        Etapa(
            nom="vs_random",
            total_timesteps=40_906,
            opponents_fn=lambda _: [RandomAgent(1)],
            mode="dsac",
            hp_fn=dsac_hiperparametres_per_defecte,
            eval_opponents_fn=lambda _: [H1Agent(1)],  # avaluació periòdica contra H1Agent
        ),
        *[
            Etapa(
                nom=f"selfplay_{i}",
                total_timesteps=40_906,
                opponents_fn=lambda prev: [TrainedDSACAgent(1, carregar_model_dsac(prev, opponents=None))],
                mode="dsac",
                hp_fn=dsac_hiperparametres_per_defecte,
                eval_opponents_fn=lambda _: [H1Agent(1)],  # avaluació periòdica contra H1Agent
            )
            for i in range(1, n_selfplays + 1)
        ],
    ]
    executar_pipeline(
        etapes=pipeline,
        eval_opponents_fn=lambda _: [H1Agent(1)],
        n_eval_episodes=N_EVAL_EPISODES_FINAL,
    )

    
if __name__ == "__main__":  
    prova_pipeline_1_ppo()
    #prova_pipeline_1_dsac()
    #prova_pipeline_2_ppo()
    #prova_pipeline_2_dsac()
    #prova_pipeline_3_ppo()
    #prova_pipeline_3_dsac()
    #prova_pipeline_4_ppo() 
    #prova_pipeline_4_dsac()
