from discretize import action_random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

Policy = Callable[[np.ndarray], int]
TrainingPolicy = Callable[[np.ndarray, int, int], int]


def to_epsilon_greedy(policy: Policy, epsilon: Callable[[int], float]) -> TrainingPolicy:
    def training_policy(state: np.ndarray, episode: int, max_episode: int) -> int:
        if np.random.random() <= epsilon(episode):
            # Take random action
            return action_random()
        else:
            return policy(state)
    return training_policy


def from_model(model: nn.Module) -> Policy:
    def policy(state: np.ndarray) -> int:
        state_tensor = torch.transpose(torch.from_numpy(state), 0, -1).float()
        ordered_q_values = model(state_tensor.unsqueeze(0)).argsort(descending=True)[0]
        return ordered_q_values[0].item()
    return policy
