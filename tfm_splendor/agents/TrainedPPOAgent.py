from splendor.template import Agent
from splendor.splendor.gym.envs.utils import create_action_mapping, create_legal_actions_mask

from splendor.template import Agent
from splendor.splendor import features

import numpy as np


class TrainedPPOAgent(Agent):
    def __init__(self, _id, model, deterministic=True):
        super().__init__(_id)
        self.model = model
        self.deterministic = deterministic

    def SelectAction(self, available_actions, game_state, game_rule):
        obs = features.extract_metrics_with_cards(game_state, self.id).astype(np.float32)

        action_mask = create_legal_actions_mask(
            available_actions,
            game_state,
            self.id,
        ).astype(bool)

        mapping = create_action_mapping(
            available_actions,
            game_state,
            self.id,
        )

        action_idx, _ = self.model.predict(
            obs,
            deterministic=self.deterministic,
            action_masks=action_mask,
        )
        action_idx = int(action_idx)

        if not action_mask[action_idx]:
            valid_idx = np.flatnonzero(action_mask)
            action_idx = int(valid_idx[0])

        return mapping[action_idx]    