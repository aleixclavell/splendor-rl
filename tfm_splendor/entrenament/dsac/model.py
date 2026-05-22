from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from tianshou.policy import DiscreteSACPolicy


class DSACModel:
    """Contenidor lleuger del model DSAC amb predict/save per avaluacio i inferencia."""

    def __init__(
        self,
        policy: DiscreteSACPolicy,
        hp: dict[str, Any],
        apply_action_mask_fn: Callable[[torch.Tensor, np.ndarray | torch.Tensor | None], torch.Tensor],
    ):
        self.policy = policy
        self.hp = hp
        self.device = hp["device"]
        self._apply_action_mask_fn = apply_action_mask_fn

    @torch.no_grad()
    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        action_masks: np.ndarray | None = None,
    ):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logits, _ = self.policy.actor(obs_t)
        logits = self._apply_action_mask_fn(logits, action_masks)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

        return int(action.item()), None

    def save(self, path: str) -> None:
        path_obj = Path(path)
        if path_obj.suffix == "":
            path_obj = path_obj.with_suffix(".pth")
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "hyperparameters": self.hp,
            },
            path_obj,
        )
