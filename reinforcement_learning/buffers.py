"""Rollout and PPO buffers for PPO training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch


class RolloutBuffer:
    """Store transitions collected from an environment."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.obs: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

        if len(self.obs) > self.capacity:
            self.obs.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.log_probs.pop(0)
            self.dones.pop(0)

    def sample(self) -> dict[str, torch.Tensor]:
        obs = torch.cat(self.obs, dim=0)
        actions = torch.cat(self.actions, dim=0)
        rewards = torch.cat(self.rewards, dim=0)
        values = torch.cat(self.values, dim=0)
        log_probs = torch.cat(self.log_probs, dim=0)
        dones = torch.cat(self.dones, dim=0)
        return {
            "obs": obs,
            "action": actions,
            "reward": rewards,
            "value": values,
            "log_prob": log_probs,
            "done": dones,
        }


@dataclass
class PPOBuffer:
    """Buffer container for PPO training batch data."""

    obs: List[torch.Tensor] = field(default_factory=list)
    actions: List[torch.Tensor] = field(default_factory=list)
    rewards: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    dones: List[torch.Tensor] = field(default_factory=list)
    _advantages: List[torch.Tensor] = field(default_factory=list)
    returns: List[torch.Tensor] = field(default_factory=list)

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: torch.Tensor,
        advantage: torch.Tensor,
        return_value: torch.Tensor,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self._advantages.append(advantage)
        self.returns.append(return_value)

    @property
    def advantages(self) -> torch.Tensor:
        if not self._advantages:
            raise ValueError("No advantages stored")
        advantages = torch.cat(self._advantages, dim=0)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return advantages
