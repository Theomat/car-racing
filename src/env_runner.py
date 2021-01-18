from typing import List, Any, Tuple
from discretize import action_discrete2continous
from policies import Policy, TrainingPolicy

import numpy as np
import torch

import gym


Transition = Tuple[torch.FloatTensor, int, float, Any, torch.FloatTensor]
Episode = List[Transition]

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def __rgb2gray(observation: np.ndarray) -> np.ndarray:
    gray = 0.2989 * observation[:, :, 0] + 0.5870 * observation[:, :, 1] + 0.1140 * observation[:, :, 2]
    gray = np.uint8(gray)
    return gray


def __shift_add_tensor(state: torch.FloatTensor, new_tensor: torch.FloatTensor):
    state = torch.roll(state, -1, 0)
    state[-1, :, :] = new_tensor
    return state


def __to_torch(observation: np.ndarray) -> torch.FloatTensor:
    return torch.from_numpy(observation.copy()).float()


def run_episodes(policy: TrainingPolicy, episodes: int, max_steps: int = 10000,
                 render: bool = False, frames_stack: int = 4) -> List[Episode]:
    """
    Run a certain number of episodes given the specific policy.

    Parameters:
    ============
    - **policy**: a (state, episode, max_episode) -> action mapping
    - **episodes**: the number of episodes to run
    - **max_steps**: the maximum number of steps allowed within the environment
    - **render**: whether to render the environment or not
    - **frames_stack**: the number of frames stacked for a state

    Return:
    ============
    The list of transition data for each episode.
    """
    env = gym.make('CarRacing-v0')
    episodic_data: List[Episode] = []
    state: torch.FloatTensor = torch.zeros((frames_stack, 96, 96), dtype=torch.float).to(device)
    for i_episode in range(episodes):
        observation: np.ndarray = env.reset()
        torch_observation: torch.FloatTensor = __to_torch(__rgb2gray(observation))
        for i in range(frames_stack):
            state[i, :, :] = torch_observation.clone()
        episode_data: Episode = []
        for t in range(max_steps):
            if render:
                env.render()
            discrete_action: int = policy(state, i_episode, episodes)
            observation, reward, done, info = env.step(action_discrete2continous(discrete_action))
            new_state: torch.FloatTensor = __shift_add_tensor(state.clone(), __to_torch(__rgb2gray(observation)))
            episode_data.append((state, discrete_action, reward, info, new_state))
            state: torch.FloatTensor = new_state
            if done:
                break
        episodic_data.append(episode_data)
    env.close()
    return episodic_data


def evaluate(policy: Policy, episodes: int, max_steps: int = 10000,
             render: bool = False, frames_stack: int = 4) -> List[float]:
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
        torch_observation: torch.FloatTensor = __to_torch(observation).to(device)
        for i in range(frames_stack):
            state[i, :, :] = torch_observation.clone()
        episode_reward: float = 0
        for t in range(max_steps):
            if render:
                env.render()
            discrete_action: int = policy(state)
            observation, reward, done, info = env.step(action_discrete2continous(discrete_action))
            new_state: torch.FloatTensor = __shift_add_tensor(state.clone(), __to_torch(__rgb2gray(observation)))
            episode_reward += reward
            state: torch.FloatTensor = new_state
            if done:
                break
        rewards.append(episode_reward)
    env.close()
    return rewards
