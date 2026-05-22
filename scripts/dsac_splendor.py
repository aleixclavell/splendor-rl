"""Entrenament de Splendor amb Discrete SAC (Tianshou), amb action masking.

Objectiu:
- tenir un script separat de PPO
- prioritzar que l'estructura sigui coherent i fàcil d'adaptar després
- deixar la màscara per a una segona iteració

Notes importants:
- Aquest script està escrit pensant en l'API clàssica de Tianshou 1.1.x.
- Si utilitzes Tianshou 2.x, l'API del trainer ha canviat.
- Si l'observació de l'entorn NO és un vector/Box pla, caldrà adaptar `state_shape`.
"""

from __future__ import annotations

import os
import random
import json
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
from tqdm import tqdm
import numpy as np
import torch
import splendor.splendor.gym  # noqa: F401 - registra l'entorn
from torch.utils.tensorboard import SummaryWriter

from codecarbon import EmissionsTracker
from splendor.agents.generic.random import RandomAgent

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

from tfm_splendor.entrenament.artifact_utils import preparar_execucio, enviar_mail, info_equip
from tfm_splendor.entrenament.envs import make_env_dsac
from tfm_splendor.entrenament.utils import read_codecarbon_csv
from tfm_splendor.entrenament.custom_wrappers import splendor_winner_agent_ids
from tfm_splendor.entrenament.evaluation_utils import avaluar_model
from tfm_splendor.entrenament.dsac.model import DSACModel
from tfm_splendor.entrenament.dsac.policy import MaskedDiscreteSACPolicy,apply_action_mask
from tfm_splendor.agents.H1Agent import H1Agent
from tfm_splendor.entrenament.tuning.dsac import optuna_study


from tfm_splendor.entrenament.config import dsac_hiperparametres_actuals


# =========================
# Configuració i utilitats
# =========================


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


N_EVAL_TIMESTEPS_TRAIN = 100_000
N_EVAL_EPISODES_PERIODIC = 25
N_EVAL_EPISODES_FINAL = 200
N_BEST_MODEL_EVAL_EPISODES = 50

def llista_oponents(): return [H1Agent(0)]


def mostrejar_accions_valides(
    opponents: list[Any] | None = None,
    n_episodes: int = 100,
    output_csv: str = "accions_valides.csv",
) -> dict:
    """Executa episodis complets i registra quantes accions vàlides hi ha a cada pas.

    Desa un CSV amb (episodi, pas, n_accions_valides) i imprimeix estadístiques.
    Útil per calibrar target_entropy d'auto_alpha.
    """
    if opponents is None:
        opponents = llista_oponents()

    env = make_env_dsac(opponents, monitor_file=None, flatten_obs=True, for_training=True)
    counts: list[int] = []

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["episodi", "pas", "n_accions_valides"])
        f.flush()

        for ep in range(n_episodes):
            _, info = env.reset()
            step = 0
            done = False
            while not done:
                mask = info.get("action_mask")
                n_valid = int(np.sum(mask)) if mask is not None else -1
                counts.append(n_valid)
                writer_csv.writerow([ep, step, n_valid])
                valid_actions = np.where(mask)[0] if mask is not None else np.arange(env.action_space.n)
                action = int(np.random.choice(valid_actions))
                _, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

    env.close()

    arr = np.array([c for c in counts if c > 0])
    stats = {
        "n_passos": len(arr),
        "mitjana": float(np.mean(arr)),
        "mediana": float(np.median(arr)),
        "maxim": int(np.max(arr)),
        "minim": int(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }
    print(f"\n=== Distribució d'accions vàlides ({n_episodes} episodis, {len(arr)} passos) ===")
    print(f"  Mitjana:  {stats['mitjana']:.1f}")
    print(f"  Mediana:  {stats['mediana']:.1f}")
    print(f"  Màxim:    {stats['maxim']}")
    print(f"  Mínim:    {stats['minim']}")
    print(f"  P25/P75:  {stats['p25']:.1f} / {stats['p75']:.1f}")
    print(f"  P95:      {stats['p95']:.1f}")
    print(f"  target_entropy recomanada (0.98*log(mitjana)): {0.98 * np.log(stats['mitjana']):.3f}")
    print(f"  CSV desat a: {output_csv}\n")
    return stats


def _log_codecarbon_to_tensorboard(
    writer: SummaryWriter,
    emissions_csv_path: str,
    emissions_total_kg: float | None,
    global_step: int,
) -> None:
    if emissions_total_kg is not None:
        writer.add_scalar("codecarbon/emissions_kg_total", emissions_total_kg, global_step)

    metrics = read_codecarbon_csv(emissions_csv_path)
    key_to_tag = {
        "energy_consumed": "codecarbon/energy_kwh_total",
        "cpu_energy": "codecarbon/energy_kwh_cpu",
        "gpu_energy": "codecarbon/energy_kwh_gpu",
        "ram_energy": "codecarbon/energy_kwh_ram",
        "cpu_power": "codecarbon/power_w_cpu",
        "gpu_power": "codecarbon/power_w_gpu",
        "ram_power": "codecarbon/power_w_ram",
        "emissions_rate": "codecarbon/emissions_rate",
    }
    for key, tag in key_to_tag.items():
        value = metrics.get(key)
        if value is not None:
            writer.add_scalar(tag, value, global_step)
    writer.flush()


def _rollout_stats_from_monitor(monitor_dir: str | None, last_n: int = 100) -> dict[str, float]:
    """Llegeix les últimes `last_n` episodis dels fitxers Monitor de train.

    Equivalent a ep_info_buffer de SB3: dona recompensa i longitud recent de l'entrenament.
    """
    if not monitor_dir or not Path(monitor_dir).exists():
        return {}
    rewards: list[float] = []
    lengths: list[float] = []
    for f in Path(monitor_dir).glob("*.monitor.csv"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                lines = [ln for ln in fh.readlines()[2:] if ln.strip()]
            for line in lines[-last_n:]:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    rewards.append(float(parts[0]))
                    lengths.append(float(parts[1]))
        except Exception:
            pass
    if not rewards:
        return {}
    return {
        "ep_rew_mean":   float(np.mean(rewards)),
        "ep_rew_std":    float(np.std(rewards)),
        "ep_rew_median": float(np.median(rewards)),
        "ep_len_mean":   float(np.mean(lengths)),
        "ep_len_std":    float(np.std(lengths)),
        "ep_count":      float(len(rewards)),
    }


# =========================
# Xarxes i model
# =========================


def _build_dsac_model(hp: dict, opponents: list[Any]) -> "DSACModel":
    """Construeix el model DSAC complet (actor, crítics, política) a partir d'hp."""
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
        preprocess_net=actor_preprocess, action_shape=action_shape,
        hidden_sizes=(), softmax_output=False,
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=hp["actor_lr"])

    critic1_preprocess = Net(state_shape=state_shape, hidden_sizes=hp["hidden_sizes"])
    critic1 = Critic(
        preprocess_net=critic1_preprocess, hidden_sizes=(), last_size=action_shape,
    ).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=hp["critic_lr"])

    critic2_preprocess = Net(state_shape=state_shape, hidden_sizes=hp["hidden_sizes"])
    critic2 = Critic(
        preprocess_net=critic2_preprocess, hidden_sizes=(), last_size=action_shape,
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


def generar_model_discrete_sac(agents: list[Any] | None = None, hp_fn=None):
    if agents is None:
        agents = [RandomAgent(1)]
    hp = (hp_fn or dsac_hiperparametres_actuals)()
    set_seed(hp["seed"])
    model = _build_dsac_model(hp, agents)
    return model, hp


def reset_critic_dsac(model: "DSACModel") -> None:
    """Reinicialitza els critics de DSAC conservant l'actor i alpha.

    Reinicialitza:
      - policy.critic  / policy.critic2        (Q-nets principals)
      - policy.critic_old / policy.critic2_old  (target nets → sincronitzades)
    Esborra l'estat dels optimitzadors de critic per descartar els moments
    acumulats de l'oponent anterior. L'optimitzador de l'actor no es toca.
    """
    policy = model.policy

    def _reset_module(module: torch.nn.Module) -> None:
        for layer in module.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    # Reinicialitza Q-nets principals
    _reset_module(policy.critic)
    _reset_module(policy.critic2)

    # Sincronitza targets amb hard update (τ=1) per evitar inconsistències
    policy.critic_old.load_state_dict(policy.critic.state_dict())
    policy.critic2_old.load_state_dict(policy.critic2.state_dict())

    # Esborra l'estat d'Adam dels optimitzadors de critic
    policy.critic_optim.state.clear()
    policy.critic2_optim.state.clear()

    print("  → Critics DSAC reinicialitzats (actor i alpha conservats).")


# =========================
# Entrenament i avaluació
# =========================


def evaluate_winrate(model: DSACModel, env, n_episodes: int = 20):
    wins = 0
    rewards = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        my_id = info.get("my_id", 0)
        action_masks = info.get("action_mask")
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, terminated, truncated, info = env.step(int(action))
            action_masks = info.get("action_mask")
            done = terminated or truncated

        agents = env.unwrapped.game_rule.current_game_state.agents
        puntuacio_agent = agents[my_id].score
        rewards.append(puntuacio_agent)
        if my_id in splendor_winner_agent_ids(agents):
            wins += 1

    return wins / n_episodes, float(np.mean(rewards))


def entrenar_dsac(
    opponents: list[Any] | None = None,
    total_timesteps: int = 50_000,
    model: "DSACModel | None" = None,
    model_path: str | None = None,
    emissions_file: str | None = None,
    train_monitor_dir: str | None = None,
    test_monitor_dir: str | None = None,
    logger_dir: str | None = None,
    hp_fn=None,
    config_dir: str | None = None,
    eval_opponents: list[Any] | None = None,
    initial_model_path: str | None = None,
):
    if opponents is None:
        opponents = llista_oponents()
    if eval_opponents is None:
        eval_opponents = opponents

    # Generar model i hiperparàmetres (o reutilitzar un model existent)
    if model is None:
        model, hp = generar_model_discrete_sac(agents=opponents, hp_fn=hp_fn)
    else:
        hp = model.hp

    if config_dir is not None:
        Path(config_dir).mkdir(parents=True, exist_ok=True)
        hp_to_save = dict(hp)
        hp_to_save["opponent_agents"] = ", ".join(type(op).__name__ for op in opponents)
        hp_to_save["total_timesteps"] = total_timesteps
        hp_to_save["initial_model_path"] = initial_model_path
        hp_to_save.update(info_equip())
        sufix = "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        hp_path = Path(config_dir) / f"hiperparametres{sufix}.json"
        with open(hp_path, "w", encoding="utf-8") as f:
            json.dump(hp_to_save, f, indent=4)

    # Crear directoris per monitors si cal
    if train_monitor_dir is not None:
        Path(train_monitor_dir).mkdir(parents=True, exist_ok=True)
    if test_monitor_dir is not None:
        Path(test_monitor_dir).mkdir(parents=True, exist_ok=True)

    def build_train_env(i: int):
        monitor_file = None
        if train_monitor_dir is not None:
            monitor_file = str(Path(train_monitor_dir) / f"train_env_{i}")
        return make_env_dsac(opponents, monitor_file=monitor_file, flatten_obs=True, for_training=True)

    train_envs = DummyVectorEnv(
        [lambda idx=i: build_train_env(idx) for i in range(hp["training_num"])]
    )

    train_collector = Collector(
        model.policy,
        train_envs,
        VectorReplayBuffer(hp["buffer_size"], hp["training_num"]),
    )

    effective_step_per_epoch = min(hp["step_per_epoch"], total_timesteps)
    env_num = hp["training_num"]
    effective_warmup_steps = max(
        env_num,
        int(np.ceil(hp["warmup_steps"] / env_num) * env_num),
    )

    save_dir = Path(hp["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    if logger_dir is None:
        logger_dir = str(save_dir / "logs")
    Path(logger_dir).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=logger_dir)
    tensorboard_logger = TensorboardLogger(
        writer,
        train_interval=1,
        test_interval=1,
        update_interval=1,
        info_interval=1,
    )

    codecarbon_log_interval = int(hp.get("codecarbon_log_interval", 2_048))
    last_codecarbon_log_step = -codecarbon_log_interval

    def _avaluar_i_guardar(epoch: int, env_step: int) -> None:
        nonlocal best_train_eval_reward, best_eval_winrate, best_eval_train_reward

        metrics = avaluar_model(
            model,
            n_episodes=N_EVAL_EPISODES_PERIODIC,
            env=eval_env_for_fn,
            print_summary=False,
            mode="dsac",
        )
        current_winrate = float(metrics["winrate"])
        current_score = float(metrics["avg_score"])

        writer.add_scalar("eval/winrate",           current_winrate,              env_step)
        writer.add_scalar("eval/avg_score",         current_score,                env_step)
        writer.add_scalar("eval/mean_nobles",       float(metrics["avg_nobles"]), env_step)
        writer.add_scalar("eval/mean_bought_cards", float(metrics["avg_cartes"]), env_step)

        write_header = not eval_csv_path.exists()
        with open(eval_csv_path, "a", newline="", encoding="utf-8") as fcsv:
            csv_w = csv.DictWriter(fcsv, fieldnames=_eval_csv_fields)
            if write_header:
                csv_w.writeheader()
            csv_w.writerow({
                "env_step": env_step,
                "epoch": epoch,
                "eval/winrate": current_winrate,
                "eval/avg_score": current_score,
                "eval/mean_nobles": float(metrics["avg_nobles"]),
                "eval/mean_bought_cards": float(metrics["avg_cartes"]),
            })

        # Avaluació per criteri de checkpoint: contra opponents d'entrenament
        metrics_checkpoint = avaluar_model(
            model,
            n_episodes=N_BEST_MODEL_EVAL_EPISODES,
            env=best_model_env,
            print_summary=False,
            mode="dsac",
        )
        checkpoint_winrate = float(metrics_checkpoint["winrate"])
        checkpoint_train_reward = float(metrics_checkpoint["avg_train_reward"])
        writer.add_scalar("eval/checkpoint_winrate", checkpoint_winrate, env_step)
        best_train_eval_reward = max(best_train_eval_reward, checkpoint_winrate)

        is_best = (checkpoint_winrate, checkpoint_train_reward) > (best_eval_winrate, best_eval_train_reward)
        if is_best:
            best_eval_winrate = checkpoint_winrate
            best_eval_train_reward = checkpoint_train_reward
            torch.save(model.policy.state_dict(), best_model_path)
        tqdm.write(
            f"Epoch #{epoch} | step {env_step} | "
            f"winrate: {checkpoint_winrate:.3f} | train_reward: {checkpoint_train_reward:.3f} "
            f"(best winrate: {best_eval_winrate:.3f}, best train_reward: {best_eval_train_reward:.3f})"
            + (" ← nou millor model guardat" if is_best else ""),
        )

        try:
            writer.add_scalar("eval/energy_kwh_total", float(tracker._total_energy.kWh), env_step)
        except Exception:
            pass

        writer.flush()

    def train_fn(epoch: int, env_step: int) -> None:
        nonlocal last_codecarbon_log_step, last_eval_step

        # Rollout stats (cada ronda de collect, equivalent a rollout/* de PPO)
        rollout = _rollout_stats_from_monitor(train_monitor_dir)
        if rollout:
            writer.add_scalar("rollout/ep_rew_mean",   rollout["ep_rew_mean"],   env_step)
            writer.add_scalar("rollout/ep_rew_std",    rollout["ep_rew_std"],    env_step)
            writer.add_scalar("rollout/ep_rew_median", rollout["ep_rew_median"], env_step)
            writer.add_scalar("rollout/ep_len_mean",   rollout["ep_len_mean"],   env_step)
            writer.add_scalar("rollout/ep_len_std",    rollout["ep_len_std"],    env_step)
            writer.add_scalar("rollout/ep_count",      rollout["ep_count"],      env_step)
            writer.flush()

        # Alpha (temperatura d'entropia) — només quan auto_alpha és actiu
        if getattr(model.policy, "is_auto_alpha", False):
            alpha_val = model.policy.log_alpha.detach().exp().item()
            writer.add_scalar("train/alpha", alpha_val, env_step)
            writer.add_scalar("train/target_entropy", model.policy.target_entropy, env_step)
            writer.flush()

        # Avaluació periòdica (cada EVAL_STEP_INTERVAL steps)
        if env_step - last_eval_step >= EVAL_STEP_INTERVAL:
            last_eval_step = env_step
            _avaluar_i_guardar(epoch, env_step)

        # CodeCarbon (cada codecarbon_log_interval passos)
        if env_step - last_codecarbon_log_step < codecarbon_log_interval:
            return

        # flush() força l'escriptura al CSV immediatament (sense ell, CodeCarbon només escriu en stop())
        tracker.flush()

        current = read_codecarbon_csv(emissions_csv_output)
        if not current:
            return

        last_codecarbon_log_step = env_step

        emissions_total = current.get("emissions")
        energy_total = current.get("energy_consumed")

        if emissions_total is not None:
            writer.add_scalar("codecarbon/emissions_kg_total", emissions_total, env_step)

        if energy_total is not None:
            writer.add_scalar("codecarbon/energy_kwh_total", energy_total, env_step)

        map_tags = {
            "cpu_energy": "codecarbon/energy_kwh_cpu",
            "gpu_energy": "codecarbon/energy_kwh_gpu",
            "ram_energy": "codecarbon/energy_kwh_ram",
            "cpu_power": "codecarbon/power_w_cpu",
            "gpu_power": "codecarbon/power_w_gpu",
            "ram_power": "codecarbon/power_w_ram",
            "emissions_rate": "codecarbon/emissions_rate",
        }
        for key, tag in map_tags.items():
            value = current.get(key)
            if value is not None:
                writer.add_scalar(tag, value, env_step)

        writer.flush()

    # Entorn d'avaluació periòdica per a test_fn (equivalent a CallbackAvaluacioPeriodica de PPO)
    eval_env_for_fn = make_env_dsac(
        eval_opponents, monitor_file=None, flatten_obs=True,
        for_training=False, eval_loss_reward=0.0,
    )
    # Entorn per seleccionar el millor model: opponent d'entrenament, reward densa, política determinista
    best_model_env = make_env_dsac(
        opponents, monitor_file=None, flatten_obs=True, for_training=False,
    )

    EVAL_STEP_INTERVAL = 10_240
    last_eval_step = -EVAL_STEP_INTERVAL
    best_train_eval_reward = -float("inf")
    best_eval_winrate = -float("inf")
    best_eval_train_reward = -float("inf")
    best_model_path = save_dir / "dsac_best_train_eval.pth"
    eval_csv_path = Path(logger_dir) / "progress.csv"
    _eval_csv_fields = ["env_step", "epoch", "eval/winrate", "eval/avg_score", "eval/mean_nobles", "eval/mean_bought_cards"]

    # Escalfament del buffer abans de començar a actualitzar la política
    train_collector.collect(n_step=effective_warmup_steps, reset_before_collect=True)

    max_epoch = max(1, int(np.ceil(total_timesteps / effective_step_per_epoch)))

    trainer = OffpolicyTrainer(
        policy=model.policy,
        train_collector=train_collector,
        test_collector=None,
        max_epoch=max_epoch,
        step_per_epoch=effective_step_per_epoch,
        step_per_collect=hp["step_per_collect"],
        episode_per_collect=hp["episode_per_collect"],
        episode_per_test=hp["episode_per_test"],
        batch_size=hp["batch_size"],
        update_per_step=hp["update_per_step"],
        train_fn=train_fn,
        logger=tensorboard_logger,
        verbose=True,
        show_progress=True,
    )

    # CodeCarbon
    os.environ["CODECARBON_COUNTRY_ISO_CODE"] = "ESP"
    os.environ["CODECARBON_REGION"] = "catalonia"
    emissions_csv_output = emissions_file or str(save_dir / "emissions_dsac.csv")
    tracker = EmissionsTracker(
        project_name="Entrenament Splendor Discrete SAC",
        measure_power_secs=10,
        output_file=emissions_csv_output,
        allow_multiple_runs=True,
        log_level="error",
    )
    
    tracker.start()
    try:
        # Compatible amb Tianshou 1.1.x (l'API de 2.x és incompatible)
        info = trainer.run()
        # Avaluació forçada si els últims epochs no s'han avaluat
        if total_timesteps - last_eval_step >= EVAL_STEP_INTERVAL // 2:
            _avaluar_i_guardar(max_epoch, total_timesteps)
        if best_model_path.exists():
            best_state_dict = torch.load(best_model_path, map_location=hp["device"])
            model.policy.load_state_dict(best_state_dict)
        model.save(model_path or str(save_dir / "discrete_sac_splendor"))
    finally:
        emissions = tracker.stop()
        _log_codecarbon_to_tensorboard(
            writer=writer,
            emissions_csv_path=emissions_csv_output,
            emissions_total_kg=emissions,
            global_step=total_timesteps,
        )
        tensorboard_logger.finalize()
        if emissions is not None:
            print(f"Emissions totals: {emissions:.6f} kg CO2")
        else:
            print("Emissions totals: no disponibles")
        train_envs.close()
        eval_env_for_fn.close()
        best_model_env.close()

    return model, info



def carregar_model_dsac(model_path: str, opponents: list[Any] | None = None) -> DSACModel:
    """Carrega un model DSAC des d'un fitxer .pth (complet o només state_dict)."""
    checkpoint = torch.load(model_path, map_location="cpu")

    hp_fn = None
    if isinstance(checkpoint, dict) and "hyperparameters" in checkpoint:
        saved_hp = dict(checkpoint["hyperparameters"])
        saved_hp["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        hp_fn = lambda: saved_hp

    model, _ = generar_model_discrete_sac(agents=opponents, hp_fn=hp_fn)

    if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
        state_dict = checkpoint["policy_state_dict"]
    else:
        state_dict = checkpoint
    model.policy.load_state_dict(state_dict)
    return model




if __name__ == "__main__":
    print("Equip:", info_equip())
    fer_optuna = False
    if fer_optuna:
        optuna_study(n_trials=10, total_timesteps_trial=25_000)
        sys.exit(0)

    nomes_evaluar = False
    if nomes_evaluar:
        model_path_base = "runs\\dsac_20260425_124713\\model\\discrete_sac_splendor_20260425_124713.pth"
        model = carregar_model_dsac(model_path_base, opponents=llista_oponents())
        avaluar_model(model, opponents=llista_oponents(), n_episodes=10, mode="dsac")
        sys.exit(0)

    sufix = "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "dsac" + sufix
    paths = preparar_execucio(run_name)

    opponents = llista_oponents()
    total_timesteps = N_EVAL_TIMESTEPS_TRAIN

    nom_model = os.path.join(paths["model_dir"], "discrete_sac_splendor" + sufix)
    emissions_file = os.path.join(paths["emissions_dir"], "emissions_dsac.csv")
    emissions_eval_file = os.path.join(paths["emissions_dir"], "emissions_dsac_eval.csv")
    train_monitor_file = os.path.join(paths["monitor_dir"], "train")
    eval_monitor_file = os.path.join(paths["monitor_dir"], "eval_periodic")
    final_eval_monitor_file = os.path.join(paths["monitor_dir"], "eval_final")
    logger_dir = paths["logs_dir"]

    model, info = entrenar_dsac(
        opponents=opponents,
        total_timesteps=total_timesteps,
        model_path=nom_model,
        emissions_file=emissions_file,
        train_monitor_dir=train_monitor_file,
        test_monitor_dir=eval_monitor_file,
        logger_dir=logger_dir,
        config_dir=paths["config_dir"],
    )
    print("Resultat final del trainer:")
    print(info)

    tracker_eval = EmissionsTracker(
        project_name="Avaluació Final Splendor DSAC",
        measure_power_secs=10,
        output_file=emissions_eval_file,
        log_level="error",
    )
    tracker_eval.start()
    try:
        metrics = avaluar_model(
            model,
            n_episodes=N_EVAL_EPISODES_FINAL,
            opponents=opponents,
            monitor_file=final_eval_monitor_file,
            mode="dsac",
        )
    finally:
        emissions_eval = tracker_eval.stop()
        if emissions_eval is not None:
            print(f"Emissions avaluació final: {emissions_eval:.6f} kg CO2")

    enviar_mail(
        subject=f"Entrenament DSAC finalitzat - {run_name}",
        body=(
            f"L'entrenament del model DSAC al Splendor ha finalitzat.\n\n"
            f"Run: {run_name}\n"
            f"Total timesteps: {total_timesteps}\n"
            f"Model guardat a: {nom_model}\n"
            f"Monitor d'avaluació final: {final_eval_monitor_file}\n"
            f"Emissions entrenament registrades a: {emissions_file}\n"
            f"Emissions avaluació final: {emissions_eval:.6f} kg CO2\n\n"
            f"*** Avaluació final contra oponents ***\n"
            f"Winrate: {metrics['winrate']:.2%}\n"
            f"Recompensa mitjana: {metrics['avg_reward']:.3f}\n"
            f"Nobles (mitjana): {metrics.get('avg_nobles', 'N/A')}\n"
            f"Cartes comprades (mitjana): {metrics['avg_cartes']:.2f}\n\n"
        ),
    )
