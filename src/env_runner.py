from typing import List, Any, Tuple, Callable
from discretize import action_discrete2continous

import gym


Policy = Callable[[Any], int]
TrainingPolicy = Callable[[Any, int, int], int]
Transition = Tuple[Any, int, float, Any, Any]
Episode = List[Transition]


def run_episodes(policy: TrainingPolicy, episodes: int, max_steps: int = 1000, render: bool = False) -> List[Episode]:
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
        observation: Any = env.reset()
        episode_data: Episode = []
        for t in range(max_steps):
            if render:
                env.render()
            discrete_action: int = policy(observation, i_episode, episodes)
            state: Any = observation
            observation, reward, done, info = env.step(action_discrete2continous(discrete_action))
            episode_data.append((state, discrete_action, reward, info, observation))
            if done:
                break
        episodic_data.append(episode_data)
    env.close()
    return episodic_data


def evaluate(policy: Policy, episodes: int, max_steps: int = 1000, render: bool = False) -> List[float]:
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
        observation: Any = env.reset()
        episode_reward: float = 0
        for t in range(max_steps):
            if render:
                env.render()
            discrete_action: int = policy(observation)
            observation, reward, done, info = env.step(action_discrete2continous(discrete_action))
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
    env.close()
    return rewards
