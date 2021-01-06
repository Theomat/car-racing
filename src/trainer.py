import env_runner
import annealing
import discretize
from policies import TrainingPolicy, Policy, to_epsilon_greedy
from uniform_replay_buffer import UniformReplayBuffer

from typing import Callable, List

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import tqdm


def train(policy: Policy,
          epsilon: Callable[[int], float],
          replay_buffer: UniformReplayBuffer,
          episodes: int, train_frequency: int,
          test_performance_episodes: int = 0,
          max_steps: int = 10000,
          render: bool = False):
    writer: SummaryWriter = SummaryWriter()

    pbar = tqdm.pbar(episodes, desc="episodes")
    for i_training_step in range(episodes // train_frequency):
        # Produce data by interaction
        current_policy: TrainingPolicy = to_epsilon_greedy(policy,
                                                           annealing.translated(i_training_step * train_frequency),
                                                           discretize.MAX_ACTION)
        data: List[env_runner.Episode] = env_runner.run_episodes(current_policy, train_frequency, max_steps, render)
        # Log Train rewards
        train_rewards: List[float] = [sum([r for _, _, r, _, _ in episode]) for episode in episodes]
        writer.add_histogram("Reward/Test", torch.from_numpy(np.array(train_rewards)), i_training_step)
        # Store data
        replay_buffer.store(data)
        # Update progress bar
        pbar.update(train_frequency)
        # Train step
        # TODO:
        #   - actually learn
        #   - log the loss
        # Evaluation step
        if test_performance_episodes > 0:
            rewards: List[float] = env_runner.evaluate(policy, test_performance_episodes, max_steps)
            writer.add_histogram("Reward/Test", torch.from_numpy(np.array(rewards)), i_training_step)
    pbar.close()
    writer.close()
