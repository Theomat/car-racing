from typing import Tuple, List
from env_runner import Transition, Episode

import numpy as np
import torch


class UniformReplayBuffer:

    def __init__(self, size: int = 10000, seed: int = 0):
        self._size: int = size
        self._memory: List[Transition] = []
        self.generator: np.random.Generator = np.random.default_rng(seed)
        self._buffer = None

    def __convert_state__(self, state: np.ndarray):
        return torch.transpose(torch.from_numpy(state), 0, -1).float()

    def store(self, episodes: List[Episode]):
        for episode in episodes:
            for transition in episode:
                s, a, r, i, ns = transition
                if isinstance(s, np.ndarray):
                    s = self.__convert_state__(s)
                if isinstance(ns, np.ndarray):
                    ns = self.__convert_state__(ns)
                self._memory.append((s, a, r, ns))
        if len(self._memory) > self._size:
            self._memory = self._memory[-self._size:]

    def sample(self, size: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        if self._buffer is None or self._buffer[0].shape[0] != size:
            self._buffer = [torch.zeros((size, 96, 96, 3)), torch.zeros((size, 1)),
                            torch.zeros((size, 1)), torch.zeros((size, 96, 96, 3))]
        memories = self.generator.integers(0, len(self._memory), size, dtype=np.int)
        for i, index in enumerate(memories):
            sample = self._memory[index]
            self._buffer[0][i] = sample[0]
            self._buffer[1][i] = sample[1]
            self._buffer[2][i] = sample[2]
            self._buffer[3][i] = sample[3]
        return self._buffer[0], self._buffer[1], self._buffer[2], self._buffer[3]
