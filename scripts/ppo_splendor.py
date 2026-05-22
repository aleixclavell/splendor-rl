# Embolcallar l'entrenament amb codecarbon per veure el consum energètic
from datetime import datetime
import os
import sys

from codecarbon import EmissionsTracker
from splendor.agents.generic.random import RandomAgent

from sb3_contrib import MaskablePPO

from stable_baselines3.common.logger import configure

# Heurística per avaluar el winrate del model després de l'entrenament
from tfm_splendor.agents.H1Agent import H1Agent
from tfm_splendor.entrenament.envs import make_env_ppo, mask_fn
from tfm_splendor.entrenament.callbacks import CallbackAvaluacioPeriodica, CallbackEntrenament
from tfm_splendor.entrenament.config import ppo_hiperparametres_finals
from tfm_splendor.agents.TrainedPPOAgent import TrainedPPOAgent

from tfm_splendor.entrenament.evaluation_utils import avaluar_model
from tfm_splendor.entrenament.artifact_utils import enviar_mail, registrar_hiperparametres, preparar_execucio, info_equip
from tfm_splendor.entrenament.tuning.ppo import optuna_study

N_EVAL_EPISODES_PERIODIC = 100


def reset_critic_ppo(model: MaskablePPO) -> None:
    """Reinicialitza els pesos del critic (value branch) conservant l'actor.

    Reinicialitza:
      - policy.mlp_extractor.value_net  (branca shared del MLP per al valor)
      - policy.value_net                (cap final que produeix V(s))
    Elimina l'estat de l'optimitzador d'Adam per aquests paràmetres perquè
    els moments acumulats de l'etapa anterior no distorsionin la nova fase.
    """
    policy = model.policy

    critic_modules = [policy.mlp_extractor.value_net, policy.value_net]
    for module in critic_modules:
        for layer in module.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    value_param_ids = {
        id(p)
        for m in critic_modules
        for p in m.parameters()
    }
    opt_state = policy.optimizer.state
    for p in list(opt_state.keys()):
        if id(p) in value_param_ids:
            del opt_state[p]

    print("  → Critic reinicialitzat (actor conservat).")
N_EVAL_EPISODES_FINAL = 200
N_EVAL_TIMESTEPS_TRAIN = 150_000 


def llista_oponents(): return [H1Agent(1)]

def generar_model_ppo(agents=None, train_monitor_file="train_monitor"):
    if agents is None:
        agents = llista_oponents()
    train_env = make_env_ppo(agents, monitor_file=train_monitor_file, for_training=True)
    parametres = ppo_hiperparametres_finals()
    model = MaskablePPO("MlpPolicy", train_env, **parametres)
    return model, train_env

def entrenar_model_ppo(
    opponents=None,
    total_timesteps=50_000,
    nom_model="maskable_ppo",
    model_path_base=None,
    continuar_entrenant=False,
    train_monitor_file="train_monitor",
    eval_monitor_file="eval_monitor_periodic",
    logger_dir="./logs/ppo",
    hyperparams_dir="models",
    emissions_file="emissions.csv",
):
    if opponents is None:
        opponents = [H1Agent(1)]
    if continuar_entrenant:
        if not model_path_base:
            raise ValueError("Cal passar model_path_base quan continuar_entrenant=True")
        train_env = make_env_ppo(opponents, monitor_file=train_monitor_file, for_training=True)
        model = MaskablePPO.load(model_path_base, env=train_env)
    else:
        model, train_env = generar_model_ppo(agents=opponents, train_monitor_file=train_monitor_file)

    registrar_hiperparametres(
        model, hyperparams_dir,
        opponents=opponents,
        total_timesteps=total_timesteps,
        initial_model_path=model_path_base if continuar_entrenant else None,
    )

    new_logger = configure(logger_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Localització per a CodeCarbon (factor d'emissió espanyol ~0.17 kg CO2/kWh)
    os.environ["CODECARBON_COUNTRY_ISO_CODE"] = "ESP"
    os.environ["CODECARBON_REGION"] = "catalonia"

    tracker = EmissionsTracker(
        project_name="Entrenament Splendor Maskable PPO",
        measure_power_secs=10,
        output_file=emissions_file,
        log_level="error",
    )

    log_granularity = 2_048

    callbackEntrenament = CallbackEntrenament(
        tracker=tracker,
        train_monitor_file=train_monitor_file,
        verbose=0,
        emissions_csv=emissions_file,
    )
    callbackEvaluacio = CallbackAvaluacioPeriodica(
        opponent_factory=llista_oponents,
        eval_freq=log_granularity * 5,
        n_eval_episodes=N_EVAL_EPISODES_PERIODIC,
        monitor_file=eval_monitor_file,
        verbose=0,
        tracker=tracker,
    )

    tracker.start()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[callbackEntrenament, callbackEvaluacio],
            reset_num_timesteps=not continuar_entrenant,
            log_interval=1,
        )

        model.save(nom_model)
    finally:
        emissions = tracker.stop()
        print(f"Emissions totals: {emissions:.6f} kg CO2")
        train_env.close()

    return model, emissions


if __name__ == "__main__":
    print("Equip:", info_equip())

    sufix = "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    opponents = llista_oponents()
    total_timesteps = N_EVAL_TIMESTEPS_TRAIN

    continuar_entrenant = True
    model_path_base = "runs\\ppo_20260502_091718\\model\\maskable_ppo_20260502_091718.zip"

    run_name = "ppo" + sufix
    paths = preparar_execucio(run_name)

    nom_model = os.path.join(paths["model_dir"], "maskable_ppo" + sufix)
    train_monitor_file = os.path.join(paths["monitor_dir"], "train_monitor")
    eval_monitor_file = os.path.join(paths["monitor_dir"], "eval_monitor_periodic")
    final_eval_monitor_file = os.path.join(paths["monitor_dir"], "eval_final")
    logger_dir = paths["logs_dir"]
    hyperparams_dir = paths["config_dir"]
    emissions_file = os.path.join(paths["emissions_dir"], "emissions.csv")
    emissions_eval_file = os.path.join(paths["emissions_dir"], "emissions_eval.csv")

    model, emissions = entrenar_model_ppo(
        opponents=opponents,
        total_timesteps=total_timesteps,
        nom_model=nom_model,
        model_path_base=model_path_base,
        continuar_entrenant=continuar_entrenant,
        train_monitor_file=train_monitor_file,
        eval_monitor_file=eval_monitor_file,
        logger_dir=logger_dir,
        hyperparams_dir=hyperparams_dir,
        emissions_file=emissions_file,
    )

    tracker_eval = EmissionsTracker(
        project_name="Avaluació Final Splendor PPO",
        measure_power_secs=10,
        output_file=emissions_eval_file,
        log_level="error",
    )
    tracker_eval.start()
    try:
        metrics = avaluar_model(model, opponents=llista_oponents(), n_episodes=N_EVAL_EPISODES_FINAL, monitor_file=final_eval_monitor_file)
    finally:
        emissions_eval = tracker_eval.stop()
        if emissions_eval is not None:
            print(f"Emissions avaluació final: {emissions_eval:.6f} kg CO2")

    enviar_mail(
        subject=f"Entrenament PPO finalitzat - {run_name}",
        body=(
            f"L'entrenament del model PPO al Splendor ha finalitzat.\n\n"
            f"Run: {run_name}\n"
            f"Total timesteps: {total_timesteps}\n"
            f"Model guardat a: {nom_model}\n"
            f"Monitor d'avaluació final: {final_eval_monitor_file}\n"
            f"Emissions entrenament: {emissions:.6f} kg CO2\n"
            f"Emissions avaluació final: {emissions_eval:.6f} kg CO2\n\n"
            f" *** Avaluació final contra oponents *** \n"
            f"Winrate: {metrics['winrate']:.2%}\n"
            f"Recompensa mitjana: {metrics['avg_reward']:.3f}\n"
            f"Nobles (mitjana): {metrics['avg_nobles']:.2f}\n"
            f"Cartes comprades (mitjana): {metrics['avg_cartes']:.2f}\n\n"
        ),
    )

