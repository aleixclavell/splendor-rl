from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tianshou.data import Batch
from tianshou.policy import DiscreteSACPolicy


def extract_action_mask(info: Any) -> np.ndarray | None:
    # batch.info pot ser dict o objecte segons si l'entorn és vectoritzat o no
    if info is None:
        return None
    if isinstance(info, dict):
        mask = info.get("action_mask")
        return None if mask is None else np.asarray(mask, dtype=bool)
    mask = getattr(info, "action_mask", None)
    return None if mask is None else np.asarray(mask, dtype=bool)


def apply_action_mask(logits: torch.Tensor, action_mask: np.ndarray | torch.Tensor | None) -> torch.Tensor:
    if action_mask is None:
        return logits

    mask = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape != logits.shape:
        if mask.shape[0] == 1 and mask.shape[1] == logits.shape[1]:
            mask = mask.expand_as(logits)
        else:
            raise ValueError(
                f"action_mask shape {tuple(mask.shape)} does not match logits shape {tuple(logits.shape)}",
            )
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("Each state must have at least one legal action in action_mask.")

    # Usem el mínim representable del tipus en lloc de -inf per evitar NaN en el gradient
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class MaskedDiscreteSACPolicy(DiscreteSACPolicy):

    def forward(  # type: ignore
        self,
        batch,
        state: dict | Batch | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Batch:
        logits_BA, hidden_BH = self.actor(batch.obs, state=state, info=batch.info)
        action_mask = extract_action_mask(batch.info)
        # A Splendor la màscara sempre existeix; un fallback silenciós corrompria l'entrenament
        if action_mask is None:
            raise RuntimeError(f"No action_mask found. batch.info type={type(batch.info)}, value={batch.info}")
        logits_BA = apply_action_mask(logits_BA, action_mask)
        dist = torch.distributions.Categorical(logits=logits_BA)
        act_B = (
            dist.mode
            if self.deterministic_eval and not self.is_within_training_step
            else dist.sample()
        )
        return Batch(logits=logits_BA, act=act_B, state=hidden_BH, dist=dist)

    def _target_q(self, buffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]

        # Off-policy: la màscara de s' s'ha desat al buffer per WrapperObsAmbMascaraSeguent
        raw_info = batch.info
        if hasattr(raw_info, 'action_mask_next'):
            mask_next = raw_info.action_mask_next  # shape (B, n_actions)
        else:
            mask_next = None  # sense màscara és millor que usar una màscara incorrecta

        logits_next, _ = self.actor(batch.obs_next, state=None)

        if mask_next is not None:
            mask_tensor = torch.as_tensor(
                mask_next, dtype=torch.float32, device=logits_next.device
            )
            logits_next = apply_action_mask(logits_next, mask_tensor)

        dist = torch.distributions.Categorical(logits=logits_next)
        target_q = dist.probs * torch.min(
            self.critic_old(batch.obs_next),
            self.critic2_old(batch.obs_next),
        )
        # entropy calculada només sobre accions legals gràcies a la màscara aplicada als logits
        return target_q.sum(dim=-1) + self.alpha * dist.entropy()

    def process_fn(self, batch, buffer, indices):
        # Keep default SAC preprocessing so n-step returns are populated.
        return super().process_fn(batch, buffer, indices)
