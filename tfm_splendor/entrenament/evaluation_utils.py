from tfm_splendor.entrenament.envs import make_env_ppo, make_env_dsac, mask_fn
from tfm_splendor.entrenament.custom_wrappers import splendor_winner_agent_ids, splendor_bought_cards
import numpy as np


def avaluar_model(
    model,
    n_episodes=100,
    env=None,
    opponents=None,
    monitor_file="eval_monitor",
    print_summary=True,
    mode="ppo",
):
    created_env = False
    if env is None:
        if opponents is None:
            raise ValueError("Cal passar 'env' o 'opponents' per avaluar el model")
        if mode == "dsac":
            env = make_env_dsac(
                opponents=opponents,
                monitor_file=monitor_file,
                flatten_obs=True,
                for_training=False,
                eval_loss_reward=0.0,
            )
        else:
            env = make_env_ppo(opponents=opponents, monitor_file=monitor_file, for_training=False)
        created_env = True

    wins = 0
    rewards = []
    train_rewards = []
    scores = []
    nobles_comprats = []
    cartes_comprades = []
    steps_per_episode = []
    opponent_wins = {}
    tied_episodes = 0

    for _ in range(n_episodes):
        obs, info = env.reset()
        my_id = info.get("my_id", 0)  # fallback a 0 si no està present
        opponent_labels = _opponent_labels_by_id(env, my_id)
        action_masks = _get_action_mask(env, info, mode)
        done = False
        ep_reward = 0.0
        ep_train_reward = 0.0
        ep_steps = 0

        while not done:
            prediction = model.predict(obs, deterministic=True, action_masks=action_masks)
            action = prediction[0] if isinstance(prediction, tuple) else prediction
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += float(reward)
            ep_train_reward += float(info.get("train_reward", 0.0))
            action_masks = _get_action_mask(env, info, mode)
            done = terminated or truncated
            ep_steps += 1

        agents = env.unwrapped.game_rule.current_game_state.agents
        el_meu_agent = agents[my_id]

        cartes_agent = splendor_bought_cards(el_meu_agent)
        nobles_agent = len(el_meu_agent.nobles)

        rewards.append(ep_reward)
        train_rewards.append(ep_train_reward)
        scores.append(int(el_meu_agent.score))
        cartes_comprades.append(cartes_agent)
        nobles_comprats.append(nobles_agent)
        steps_per_episode.append(ep_steps)

        if ep_reward > 0:
            wins += 1

        winner_ids = splendor_winner_agent_ids(agents)
        if len(winner_ids) > 1:
            tied_episodes += 1
        for agent_id in winner_ids:
            if agent_id == my_id:
                continue
            rival_label = opponent_labels.get(agent_id, f"player_{agent_id}")
            opponent_wins[rival_label] = opponent_wins.get(rival_label, 0) + 1

    metrics = {
        "winrate": wins / n_episodes,
        "avg_reward": float(np.mean(rewards)),
        "avg_train_reward": float(np.mean(train_rewards)),
        "avg_score": float(np.mean(scores)),
        "avg_nobles": float(np.mean(nobles_comprats)),
        "avg_cartes": float(np.mean(cartes_comprades)),
        "avg_steps": float(np.mean(steps_per_episode)),
        "opponent_wins": opponent_wins,
        "tied_episodes": tied_episodes,
    }

    if print_summary:
        opponent_txt = [type(op).__name__ for op in opponents] if opponents is not None else "env extern"
        print(
            f"Winrate: {metrics['winrate']:.2%}  |  "
            f"Recompensa mitjana: {metrics['avg_reward']:.3f}  |  "
            f"Nobles (mitjana): {metrics['avg_nobles']:.2f}  |  "
            f"Cartes comprades (mitjana): {metrics['avg_cartes']:.2f}  |  "
            f"Passos (mitjana): {metrics['avg_steps']:.1f}  |  "
            f"Oponent: {opponent_txt} |  Episodis: {n_episodes}"
        )
        if opponent_wins:
            rival_win_text = " | ".join(
                f"{name} wins {count/n_episodes:.2%}"
                for name, count in sorted(opponent_wins.items())
            )
            print(f"Victòries rivals: {rival_win_text}")
        if tied_episodes:
            print(f"Episodis empatats: {tied_episodes/n_episodes:.2%}")

    if created_env:
        env.close()

    return metrics


def avaluar_reward_entrenament(model, env, n_episodes=50, mode="dsac"):
    """Avalua el model amb política determinista trackejant la reward acumulada per episodi."""
    rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        action_masks = _get_action_mask(env, info, mode)
        done = False
        ep_reward = 0.0
        while not done:
            prediction = model.predict(obs, deterministic=True, action_masks=action_masks)
            action = prediction[0] if isinstance(prediction, tuple) else prediction
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += float(reward)
            action_masks = _get_action_mask(env, info, mode)
            done = terminated or truncated
        rewards.append(ep_reward)
    return float(np.mean(rewards))


def _get_action_mask(env, info, mode):
    if mode == "dsac" and isinstance(info, dict):
        mask = info.get("action_mask")
        if mask is not None:
            return np.asarray(mask, dtype=bool)
    return mask_fn(env)


def _opponent_labels_by_id(env, my_id):
    labels = {}
    for agent in env.unwrapped.agents:
        agent_id = getattr(agent, "id", None)
        if agent_id is None or agent_id == my_id:
            continue
        labels[agent_id] = type(agent).__name__
    return labels