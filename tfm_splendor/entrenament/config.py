import torch


def ppo_hiperparametres_per_defecte():
    """
    Hiperparàmetres habituals de PPO.

    Notes generals:
    - Els rangs són orientatius. No hi ha un valor òptim universal.
    - Els valors adequats depenen de l'entorn, la mida de l'espai d'accions
      i l'estabilitat de l'entrenament.
    - Els paràmetres més sensibles: learning_rate, n_steps, batch_size,
      n_epochs, gamma i clip_range.

    Paràmetres:
    - learning_rate: velocitat d'aprenentatge. Rang: 1e-5 a 1e-3.
    - n_steps: passos recollits per actualització. Rang: 512 a 8192.
    - batch_size: mida del minibatch. Rang: 32 a 512.
    - n_epochs: passades sobre les dades recollides. Rang: 3 a 20.
    - gamma: factor de descompte. Rang: 0.95 a 0.999.
    - gae_lambda: compromís biaix-variància en GAE. Rang: 0.90 a 0.99.
    - clip_range: límit de canvi de política per actualització. Rang: 0.1 a 0.3.
    - ent_coef: pes del terme d'entropia. Rang: 0.0 a 0.02.
    - vf_coef: pes de la pèrdua del crític. Rang: 0.25 a 1.0.
    - max_grad_norm: límit del gradient per estabilitat. Rang: 0.3 a 1.0.
    - target_kl: límit de divergència KL. None o 0.01 a 0.05.
    - policy_kwargs: configuració extra de la xarxa neuronal.
    - seed: llavor aleatòria per reproduïbilitat.
    - device: "cpu", "cuda" o "auto".
    - verbose: 0, 1 o 2.
    - tensorboard_log: ruta de logs de TensorBoard.
    """
    return {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "policy_kwargs": None,
        "seed": None,
        "device": "auto",
        "verbose": 1,
        "tensorboard_log": "./ppo_splendor_tensorboard/",
    }


def ppo_hiperparametres_optuna():
    """Millors hiperparàmetres PPO de l'estudi Optuna complet (optuna_splendor_ppo)."""
    hp = ppo_hiperparametres_per_defecte()
    hp.update({
        "learning_rate": 0.0009163271364241772,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.9913165289487678,
        "gae_lambda": 0.920842046731465,
        "clip_range": 0.23384874412813772,
        "ent_coef": 0.013426149018854751,
        "vf_coef": 0.9773156039650488,
        "max_grad_norm": 0.7754106146522639,
        "tensorboard_log": "./ppo_splendor_tensorboard_optuna/",
    })
    return hp


def dsac_hiperparametres_per_defecte():
    """
    Hiperparàmetres habituals de Discrete SAC.

    - seed: llavor aleatòria per reproduïbilitat.
    - device: dispositiu de càlcul ("cpu" o "cuda").
    - actor_lr: learning rate de l'actor (política).
    - critic_lr: learning rate dels crítics Q.
    - alpha_lr: learning rate del coeficient d'entropia (si auto_alpha=True).
    - gamma: factor de descompte de recompensa futura.
    - tau: velocitat d'actualització soft de les target networks.
    - alpha: pes inicial del terme d'entropia.
    - auto_alpha: si True, alpha s'ajusta automàticament durant l'entrenament.
    - avg_valid_actions: mediana d'accions vàlides, usada per calcular target_entropy.
    - target_entropy_coef: fracció de l'entropia màxima com a target.
    - buffer_size: capacitat màxima del replay buffer.
    - batch_size: mida de minibatch per actualització.
    - hidden_sizes: arquitectura de capes ocultes (actor i critic).
    - training_num: entorns en paral·lel per recollida.
    - test_num: entorns en paral·lel per a test.
    - step_per_epoch: passos d'entorn per època.
    - episode_per_collect: episodis recollits abans de fer actualitzacions.
    - update_per_step: actualitzacions de xarxa per pas recollit.
    - eval_epoch_interval: freqüència d'avaluació (en èpoques).
    - episode_per_test: episodis d'avaluació per època.
    - max_epoch: nombre màxim d'èpoques.
    - estimation_step: n-step return per al càlcul del target Q.
    - warmup_steps: passos de recollida inicials abans d'actualitzar.
    - save_dir: directori base per a models i logs.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "seed": None,
        "device": device,
        "actor_lr": 1e-4,
        "critic_lr": 1e-3,
        "alpha_lr": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.2,
        "auto_alpha": True,
        "avg_valid_actions": 15,  # mediana empírica (min=1, med=15, mean=23, p95=78)
        "target_entropy_coef": 0.7,  # 0.98 era massa alt → política quasi-aleatòria
        "buffer_size": 100_000,
        "batch_size": 128,
        "hidden_sizes": [256, 256],
        "training_num": 4,
        "test_num": 4,
        "step_per_epoch": 2_048,
        "step_per_collect": None,
        "episode_per_collect": 8,
        "update_per_step": 0.25,
        "codecarbon_log_interval": 2_048,
        "eval_epoch_interval": 5,  # ~10240 passos, equivalent a l'interval d'avaluació de PPO
        "episode_per_test": 1,
        "max_epoch": 10,
        "estimation_step": 5,
        "warmup_steps": 2_000,
        "save_dir": "./dsac_outputs",
    }

def dsac_hiperparametres_optuna():
    """Millors hiperparàmetres DSAC trobats per Optuna v1 (winrate 0.10)."""
    return {
        "actor_lr": 0.00031954527595424065,
        "critic_lr": 0.00028287673659709337,
        "alpha_lr": 1.986136505037749e-05,
        "gamma": 0.9855695623171791,
        "tau": 0.011273026114402377,
        "batch_size": 256,
        "update_per_step": 0.27882432030473076,
        "estimation_step": 5,
        "target_entropy_coef": 0.8408457868805158,
        "warmup_steps": 2_000,
        "step_per_epoch": 5_000,
        "episode_per_collect": 4,
        "step_per_collect": None,
        "training_num": 2,
        "test_num": 2,
        "save_dir": "./dsac_optuna_tmp",
    }

