from typing import List, Any, Tuple
from discretize import action_discrete2continous
from policies import Policy, TrainingPolicy

import numpy as np

import gym


Transition = Tuple[np.ndarray, int, float, Any, np.ndarray]
Episode = List[Transition]


def run_episodes(policy: TrainingPolicy, episodes: int, max_steps: int = 10000, render: bool = False) -> List[Episode]:
    """
    Run a certain number of episodes given the specific policy.

    Parameters:
    ============
    - **policy**: a (state, episode, max_episode) -> action mapping
    - **episodes**: the number of episodes to run
    - **max_steps**: the maximum number of steps allowed within the environment
    - **render**: whether to render the environment or not

    Return:
    ============
    The list of transition data for each episode.
    """
    env = gym.make('CarRacing-v0')
    episodic_data: List[Episode] = []
    for i_episode in range(episodes):
        state: np.ndarray = env.reset().copy()
        episode_data: Episode = []
        for t in range(max_steps):
            if render:
                env.render()
            discrete_action: int = policy(state, i_episode, episodes)
            observation, reward, done, info = env.step(action_discrete2continous(discrete_action))
            new_state: np.ndarray = observation.copy()
            episode_data.append((state, discrete_action, reward, info, new_state))
            state: np.ndarray = new_state
            if done:
                break
        episodic_data.append(episode_data)
    env.close()
    return episodic_data


def evaluate(policy: Policy, episodes: int, max_steps: int = 10000, render: bool = False) -> List[float]:
    """
    Run a certain number of episodes given the specific policy to evaluate it.

    Parameters:
    ============
    - **policy**: a (state) -> action mapping
    - **episodes**: the number of episodes to run
    - **max_steps**: the maximum number of steps allowed within the environment
    - **render**: whether to render the environment or not

    Return:
    ============
    The list of each episode's total reward.
    """
    env = gym.make('CarRacing-v0')
    env.seed(0)
    rewards: List[float] = []
    for i_episode in range(episodes):
        observation: np.ndarray = env.reset()
        episode_reward: float = 0
        for t in range(max_steps):
            if render:
                env.render()
            discrete_action: int = policy(observation.copy())
            observation, reward, done, info = env.step(action_discrete2continous(discrete_action))
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
    env.close()
    return rewards
