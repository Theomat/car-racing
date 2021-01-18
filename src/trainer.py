import env_runner
import annealing
from policies import TrainingPolicy, Policy, to_epsilon_greedy
from uniform_replay_buffer import UniformReplayBuffer

from typing import Callable, List, Literal

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import tqdm


def train(policy: Policy, optimize_model: Callable[[SummaryWriter, int], Literal[None]],
          epsilon: Callable[[int], float],
          replay_buffer: UniformReplayBuffer,
          episodes: int, train_frequency: int,
          frames_stack: int = 4,
          test_performance_episodes: int = 0,
          max_steps: int = 10000,
          max_negative_steps: int = 25,
          render: bool = False):
    """
    Train given the specified policy and optimizing with the given callback.

    Parameters:
    ============
    - **policy**: the policy to use for evaluation and for epsilon greedy control
    - **optimize_model**: a callback (writer, num_training_step) -> None that should optimize the weights,
        called at each training step
    - **epsilon**: a function (episode) -> float that gives the epsilon of the episode
    - **replay_buffer**: the replay buffer to use
    - **episodes**: the number of episodes to train for
    - **train_frequency**: the number of episodes between each training step
    - **frames_stack**: number fo frames stacked for a state
    - **test_performance_episodes**: the number of episodes ot run to evaluate the performance after each training step,
        if <= 0 no test performance is tracked
    - **max_steps**: maximum number of steps allowed within the environment in each episode
    - **max_negative_steps**: maximum number of consecutive negative steps before early stopping
    - **render**: whether or not to render the environment
    """
    writer: SummaryWriter = SummaryWriter()

    pbar = tqdm.tqdm(total=episodes, desc="episodes")
    for i_training_step in range(episodes // train_frequency):
        # Produce data by interaction
        current_policy: TrainingPolicy = to_epsilon_greedy(policy,
                                                           annealing.translated(i_training_step * train_frequency, epsilon))
        data: List[env_runner.Episode] = env_runner.run_episodes(current_policy, train_frequency,
                                                                 max_steps=max_steps,
                                                                 render=render,
                                                                 frames_stack=frames_stack,
                                                                 neg_steps_early_stop=max_negative_steps)
        # Log Train rewards
        train_rewards: List[float] = [sum([r for _, _, r, _, _ in episode]) for episode in data]
        writer.add_histogram("Reward/Train", torch.from_numpy(np.array(train_rewards)), i_training_step)
        # Store data
        replay_buffer.store(data)
        # Update progress bar
        pbar.update(train_frequency)
        # Train step
        optimize_model(writer, i_training_step)
        # Evaluation step
        if test_performance_episodes > 0:
            rewards: List[float] = env_runner.evaluate(policy, test_performance_episodes, max_steps, frames_stack)
            writer.add_histogram("Reward/Test", torch.from_numpy(np.array(rewards)), i_training_step)
    pbar.close()
    writer.close()
