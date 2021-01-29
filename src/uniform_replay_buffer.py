from typing import Tuple, List
from env_runner import Transition, Episode

import numpy as np
import torch


class UniformReplayBuffer:

    def __init__(self, size: int = 10000, seed: int = 0):
        self._size: int = size
        self._memory: List[Transition] = []
        self.generator: np.random.Generator = np.random.default_rng(seed)

    def store(self, episodes: List[Episode]):
        for episode in episodes:
            for transition in episode:
                s, a, r, i, ns = transition
                self._memory.append((s, a, r, ns))
        if len(self._memory) > self._size:
            self._memory = self._memory[-self._size:]

    def clear(self):
        self._memory = []

    def sample(self, size: int) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
        memories = self.generator.integers(0, len(self._memory), size, dtype=np.int)
        states = torch.stack([self._memory[i][0] for i in memories])
        actions = torch.tensor([self._memory[i][1] for i in memories], dtype=torch.int64).view((-1, 1))
        rewards = torch.tensor([self._memory[i][2] for i in memories]).view((-1, 1))
        next_states = [self._memory[i][3] for i in memories]
        return states, actions, rewards, next_states
