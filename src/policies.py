from typing import Callable
import numpy as np

Policy = Callable[[np.ndarray], int]
TrainingPolicy = Callable[[np.ndarray, int, int], int]


def to_epsilon_greedy(policy: Policy, epsilon: Callable[[int], float], action_space: int) -> TrainingPolicy:
    def training_policy(state: np.ndarray, episode: int, max_episode: int) -> int:
        if np.random.random() <= epsilon(episode):
            # Take random action
            return np.random.randint(0, action_space)
        else:
            return policy(state)
    return training_policy
