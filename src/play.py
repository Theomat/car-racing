from typing import List, Any, Tuple
from policies import Policy, TrainingPolicy
from policy_model import PolicyModel

import numpy as np
import torch
import discretize

import gym


Transition = Tuple[torch.FloatTensor, int, float, Any, torch.FloatTensor]
Episode = List[Transition]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = PolicyModel(discretize.MAX_ACTION, 4).float().to(device)

model.load_state_dict(torch.load('./model_3000.pth'))

def __rgb2gray(observation: np.ndarray) -> np.ndarray:
    gray = 0.2989 * observation[:,:,0] + 0.5870 * observation[:,:,1] + 0.1140 * observation[:,:,2]
    #gray = np.uint8(gray)
    return gray
    #return gray[0:84, 6:90]

def __shift_add_tensor(state: torch.FloatTensor, new_tensor: torch.FloatTensor):
    state = torch.roll(state, -1, 0)
    state[3, :, :] = new_tensor
    return state


def __to_torch(observation: np.ndarray) -> torch.FloatTensor:
    return torch.from_numpy(observation.copy()).float()


def best_action(state):
    global model
    ordered_q_values = model(state.unsqueeze(0)).argsort(descending=True)[0]
    return ordered_q_values[0].item()

env = gym.make('CarRacing-v0')
state = torch.zeros((4, 96, 96), dtype=torch.float).to(device)
observation: np.ndarray = env.reset()
torch_observation: torch.FloatTensor = __to_torch(__rgb2gray(observation))
for i in range(4):
    state[i, :, :] = torch_observation.clone()
done = False
while not done:

    env.render()
    discrete_action = best_action(state)
    print(discrete_action)
    observation, reward, done, info = env.step(discretize.action_discrete2continous(discrete_action))
    state = __shift_add_tensor(state.clone(), __to_torch(__rgb2gray(observation)))

env.close()
