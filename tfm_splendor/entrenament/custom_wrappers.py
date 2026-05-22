import gymnasium as gym
import numpy as np



def splendor_bought_cards(agent):
    return sum(len(cards) for color, cards in agent.cards.items() if color != "yellow")


def splendor_winner_agent_ids(agents):
    max_score = max(agent.score for agent in agents)
    top_ids = [index for index, agent in enumerate(agents) if agent.score == max_score]

    min_cards = min(splendor_bought_cards(agents[index]) for index in top_ids)
    return [index for index in top_ids if splendor_bought_cards(agents[index]) == min_cards]



def splendor_did_win(agents, my_id):
    """Retorna True si l'agent principal guanya segons regla oficial."""
    me = agents[my_id]

    rivals = [a for a in agents if a.id != my_id]
    if not rivals:
        return True

    my_score = me.score
    max_rival_score = max(a.score for a in rivals)

    if my_score > max_rival_score:
        return True
    if my_score < max_rival_score:
        return False

    # Empat a puntuació: desempat per menys cartes comprades.
    tied = [a for a in agents if a.score == my_score]
    my_cards = splendor_bought_cards(me)
    min_cards = min(splendor_bought_cards(a) for a in tied)

    if my_cards <= min_cards:
        return True
    return False


def mask_fn(env):
    return env.unwrapped.get_legal_actions_mask().astype(bool)


def _compute_train_reward(base_reward, done, did_win, win_bonus, loss_penalty, step_penalty):
    r = float(base_reward) * 0.1 - step_penalty
    if done:
        r += win_bonus if did_win else loss_penalty
    return r


class WrapperRecompensaEntrenament(gym.Wrapper):
    def __init__(self, env, win_bonus=1.0, loss_penalty=-1.0, step_penalty=0.005):
        super().__init__(env)
        self.win_bonus = win_bonus
        self.loss_penalty = loss_penalty
        self.step_penalty = step_penalty
        self.my_id = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.my_id = info.get("my_id", 0)
        return obs, info

    def _did_win(self):
        agents = self.unwrapped.game_rule.current_game_state.agents
        return splendor_did_win(agents, self.my_id)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        shaped_reward = _compute_train_reward(
            reward, done, self._did_win() if done else False,
            self.win_bonus, self.loss_penalty, self.step_penalty,
        )
        return obs, shaped_reward, terminated, truncated, info



class WrapperResultatAvaluacio(gym.Wrapper):
    """Wrapper d'avaluacio: recompensa de resultat d'episodi.

    - win_reward: recompensa si l'agent guanya
    - loss_reward: recompensa si l'agent perd

    Durant l'episodi retorna 0.0, i al final retorna la recompensa de resultat.
    Si es passen train_win_bonus/train_loss_penalty/train_step_penalty, afegeix
    info["train_reward"] a cada step amb el reward d'entrenament equivalent.
    """

    def __init__(self, env, win_reward=1.0, loss_reward=0.0,
                 train_win_bonus=None, train_loss_penalty=None, train_step_penalty=None):
        super().__init__(env)
        self.win_reward = float(win_reward)
        self.loss_reward = float(loss_reward)
        self.train_win_bonus = train_win_bonus
        self.train_loss_penalty = train_loss_penalty
        self.train_step_penalty = train_step_penalty
        self._track_train_reward = (
            train_win_bonus is not None
            and train_loss_penalty is not None
            and train_step_penalty is not None
        )
        self.my_id = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.my_id = info.get("my_id", 0)
        return obs, info

    def _did_win(self):
        agents = self.unwrapped.game_rule.current_game_state.agents
        return splendor_did_win(agents, self.my_id)

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        done = terminated or truncated
        did_win = self._did_win() if done else False

        if self._track_train_reward:
            info["train_reward"] = _compute_train_reward(
                base_reward, done, did_win,
                self.train_win_bonus, self.train_loss_penalty, self.train_step_penalty,
            )

        if done:
            reward = self.win_reward if did_win else self.loss_reward
            return obs, reward, terminated, truncated, info

        return obs, 0.0, terminated, truncated, info
    

    
class WrapperInfoMascaraAccions(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["action_mask"] = mask_fn(self.unwrapped)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["action_mask"] = mask_fn(self.unwrapped)
        return obs, reward, terminated, truncated, info
    


class WrapperObsAmbMascaraSeguent(gym.Wrapper):
    """
    Afegeix 'action_mask_next' a la info de cada step.
    És la màscara de s' (l'estat on hem arribat), capturada
    ABANS de l'auto-reset, quan done=True.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            # L'entorn ja ha fet auto-reset: obs és s_0 del nou episodi.
            # La màscara de s' real és a 'final_observation' si el wrapper
            # de Gymnasium ho suporta, o bé la llegim de info["final_info"].
            # Tianshou/Gymnasium guarden l'obs final a info["final_observation"].
            final_obs = info.get("final_observation", obs)
            # Calculem màscara de l'estat terminal real
            mask_next = self._get_mask_for_obs(final_obs, terminal=True)
        else:
            mask_next = self.env.unwrapped.get_legal_actions_mask()

        info["action_mask_next"] = mask_next
        return obs, reward, terminated, truncated, info

    def _get_mask_for_obs(self, obs, terminal=False):
        if terminal:
            # A l'estat terminal totes les accions són il·legals (no hi ha s'+1)
            # Retornem màscara uniforme per no introduir biaix
            n = self.env.action_space.n
            return np.ones(n, dtype=np.float32)
        return self.env.unwrapped.get_legal_actions_mask()    