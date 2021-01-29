from typing import List, Tuple
from discretize import action_discrete2continous
from policies import Policy, TrainingPolicy

import numpy as np
import torch

import gym


Transition = Tuple[torch.FloatTensor, int, float, bool, torch.FloatTensor]
Episode = List[Transition]

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def __rgb2gray(observation: np.ndarray) -> np.ndarray:
    gray = 0.2989 * observation[:, :, 0] + 0.5870 * observation[:, :, 1] + 0.1140 * observation[:, :, 2]
    #gray = np.uint8(gray)
    return gray


def __shift_add_tensor(state: torch.FloatTensor, new_tensor: torch.FloatTensor):
    state = torch.roll(state, -1, 0)
    state[-1, :, :] = new_tensor.detach().clone()
    return state


def __to_torch(observation: np.ndarray) -> torch.FloatTensor:
    return torch.from_numpy(observation.copy()).float() / .5 - 1


def run_episodes(policy: TrainingPolicy, episodes: int, max_steps: int = 10000,
                 render: bool = False, frames_stack: int = 4,
                 neg_steps_early_stop: int = 10,
                 skip_zoom: bool = False) -> Tuple[List[Episode], List[int]]:
    """
    Run a certain number of episodes given the specific policy.

    Parameters:
    ============
    - **policy**: a (state, episode, max_episode) -> action mapping
    - **episodes**: the number of episodes to run
    - **max_steps**: the maximum number of steps allowed within the environment
    - **render**: whether to render the environment or not
    - **frames_stack**: the number of frames stacked for a state
    - **neg_steps_early_stop**: the number of consecutive negative rewards steps before early stopping
    - **skip_zoom**: skip the initial zooming

    Return:
    ============
    The list of transition data for each episode.
    """
    env = gym.make('CarRacing-v0')
    episodic_data: List[Episode] = []
    state: torch.FloatTensor = torch.zeros((frames_stack, 96, 96), dtype=torch.float).to(device)
    durations: List[int] = []
    for i_episode in range(episodes):
        observation: np.ndarray = env.reset()

        episode_data: Episode = []
        negative: int = 0
        episode_reward: int = 0

        # Try skip zoom
        if skip_zoom:
            for _ in range(50):
                observation, r, _, _ = env.step([0, 0, 0])
                episode_reward += r

        # Initialize state
        torch_observation: torch.FloatTensor = __to_torch(__rgb2gray(observation))
        for i in range(frames_stack):
            state[i, :, :] = torch_observation.detach().clone()

        for t in range(max_steps):
            if render:
                env.render()
            discrete_action: int = policy(state, i_episode, episodes)
            observation, reward, done, info = env.step(action_discrete2continous(discrete_action))

            # reward = min(reward, 1.0)
            episode_reward += reward
            if episode_reward < 0:
                durations.append(t + 1)
                break
            # Early episode stopping
            if reward < 0 and t > 300:
                negative += 1
                if negative == neg_steps_early_stop:
                    durations.append(t + 1)
                    break
            else:
                negative = 0
            if done:
                episode_data.append((state, discrete_action, reward, done, None))
                durations.append(t + 1)
                break
            # Compute new state
            new_state: torch.FloatTensor = __shift_add_tensor(state.detach().clone(), __to_torch(__rgb2gray(observation)))
            episode_data.append((state, discrete_action, reward, done, new_state))
            state: torch.FloatTensor = new_state
        if len(durations) != i_episode + 1:
            durations.append(max_steps)
        episodic_data.append(episode_data)
    env.close()
    return episodic_data, durations


def evaluate(policy: Policy, episodes: int, max_steps: int = 10000,
             render: bool = False, frames_stack: int = 4,
             skip_zoom: bool = False) -> List[float]:
    """
    Run a certain number of episodes given the specific policy to evaluate it.

    Parameters:
    ============
    - **policy**: a (state) -> action mapping
    - **episodes**: the number of episodes to run
    - **max_steps**: the maximum number of steps allowed within the environment
    - **render**: whether to render the environment or not
    - **frames_stack**: the number of frames stacked for a state

    Return:
    ============
    The list of each episode's total reward.
    """
    env = gym.make('CarRacing-v0')
    env.seed(0)
    rewards: List[float] = []
    state: torch.FloatTensor = torch.zeros((frames_stack, 96, 96), dtype=torch.float).to(device)
    for i_episode in range(episodes):
        observation: np.ndarray = env.reset()
        # Try skip zoom
        if skip_zoom:
            for _ in range(50):
                observation, _, _, _ = env.step([0, 0, 0])

        # Initialize state
        torch_observation: torch.FloatTensor = __to_torch(__rgb2gray(observation))
        for i in range(frames_stack):
            state[i, :, :] = torch_observation.detach().clone()
        episode_reward: float = 0
        for t in range(max_steps):
            if render:
                env.render()
            discrete_action: int = policy(state)
            observation, reward, done, info = env.step(action_discrete2continous(discrete_action))
            new_state: torch.FloatTensor = __shift_add_tensor(state.detach().clone(), __to_torch(__rgb2gray(observation)))
            episode_reward += reward
            state: torch.FloatTensor = new_state
            if done:
                break
        rewards.append(episode_reward)
    env.close()
    return rewards
