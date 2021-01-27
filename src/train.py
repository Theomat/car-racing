import gym
import cv2
import numpy as np
import random
import math

from dqn import DQN
from replay_memory import ReplayMemory, Transition
from actions import discrete2cont, NOTHING, ACTION_SPACE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES = 5000
RENDER = False
SKIP_ZOOM = True
FRAME_STACK = 3
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
MAX_NEG_STEPS = 80
SAVE_EVERY = 1000
UPDATE_EVERY = 4
EARLY_STOP = True
MEMORY_SIZE = 10000

EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 1000
CLOSE_EVERY = 2

env = gym.make('CarRacing-v0')

policy_net = DQN(FRAME_STACK, 84, ACTION_SPACE).double().to(device)


memory = ReplayMemory(MEMORY_SIZE)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
writer = SummaryWriter()


steps_done = 0
t_step = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(ACTION_SPACE)]], device=device, dtype=torch.long)

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.CenterCrop((84, 84)),
                                transforms.Grayscale(),
                                transforms.ToTensor()])
def torchfy(obs):
    return transform(obs[:84, :, :]).to(device).clone().squeeze(0)

def optimize_model(call):
    if len(memory) < BATCH_SIZE:
        print('hi')
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    state_batch = torch.stack(batch.state)
    next_state_batch = torch.stack(batch.next_state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = policy_net(next_state_batch).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    writer.add_scalar("Loss/Train", loss.item(), call)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

opt_call = 0
for episode in tqdm(range(EPISODES)):

    done = False
    observation = env.reset()

    if SKIP_ZOOM:
        for _ in range(50):
            observation, _, _, _ = env.step(NOTHING)
            RENDER and env.render()

    state = torch.zeros([FRAME_STACK, 84, 84], dtype=torch.float64, device=device).double()

    for i in range(FRAME_STACK):
        state[i, :, :] = torchfy(observation)

    score = 0
    negative = 0
    step_counter = 0
    while not done:

        action = select_action(state)

        observation, reward, done, _ = env.step(discrete2cont(action.item()))

        # reward = min(reward, 1.0)
        score += reward

        # Early episode stopping
        if EARLY_STOP:
            if reward < 0 and step_counter > 300:
                negative += 1
                if negative == MAX_NEG_STEPS:
                    done = True
            elif reward > 0:
                negative = 0

        reward = torch.tensor([reward], device=device)

        prev_state = state.clone()

        state = torch.roll(state, -1, 0)
        state[-1, :, :] = torchfy(observation)

        memory.push(prev_state, action, state, reward)

        t_step = (t_step + 1) % UPDATE_EVERY
        if t_step == 0:
            optimize_model(opt_call)
            opt_call += 1


        RENDER and env.render()
        step_counter += 1

    if episode % CLOSE_EVERY == 0:
        env.close()

    if episode % SAVE_EVERY == 0:
        torch.save(policy_net.state_dict(), './model_{}.pth'.format(episode))

    writer.add_scalar("Reward/Train", score, episode)
    writer.add_scalar("Duration/Train", step_counter, episode)

    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    writer.add_scalar("Epsilon", eps_threshold, episode)

    steps_done += 1


env.close()
torch.save(policy_net.state_dict(), './model_latest.pth')
