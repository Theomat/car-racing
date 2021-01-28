import gym
import cv2
import numpy as np
import random
import math

from dqn import DQN
from replay_memory import ReplayMemory
from actions import discrete2cont, NOTHING, ACTION_SPACE

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from torchvision import transforms

from tqdm import tqdm


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

optimizer = Adam(lr=LR, epsilon=1e-7)
policy_net = DQN(FRAME_STACK, 84, ACTION_SPACE)
policy_net.compile(loss='mean_squared_error', optimizer=optimizer)


memory = ReplayMemory(MEMORY_SIZE)

writer = SummaryWriter()


steps_done = 0
t_step = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

    if sample > eps_threshold:
        prediction = policy_net.predict(np.expand_dims(state, axis=0))[0]
        return np.argmax(prediction)
    else:
        return random.randrange(ACTION_SPACE)


transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.CenterCrop((84, 84)),
                                transforms.Grayscale(),
                                transforms.ToTensor()])



def torchfy(obs):
    #TODO grayscale
    return obs[:84, :84, :].copy() / 255.0


def optimize_model(call):
    if len(memory) < BATCH_SIZE:
        print('hi')
        return
    transitions = memory.sample(BATCH_SIZE)
    states = np.stack([s for s, _, _, _ in transitions])
    next_states = np.stack([ns for _, _, _, ns in transitions if ns is not None])
    next_state_values = policy_net.predict(next_states).amax(1)
    target_values = policy_net.predict(states)
    next_state_index = 0
    for i, transition in enumerate(transitions):
        s, a, r, ns = transition
        if ns is None:
            target_values[i, a] = r
        else:
            target_values[i, a] = r + GAMMA * next_state_values[next_state_index]
            next_state_index += 1

    policy_net.fit(states, target_values, epochs=1, verbose=0)


opt_call = 0
for episode in tqdm(range(EPISODES)):

    done = False
    observation = env.reset()

    if SKIP_ZOOM:
        for _ in range(50):
            observation, _, _, _ = env.step(NOTHING)
            RENDER and env.render()

    state = np.zeros((FRAME_STACK, 84, 84), dtype=np.float)

    for i in range(FRAME_STACK):
        state[i, :, :] = torchfy(observation)

    score = 0
    negative = 0
    step_counter = 0
    while not done:

        action = select_action(state)

        observation, reward, done, _ = env.step(discrete2cont(action))

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
        prev_state = state.copy()

        # TODO
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
        policy_net.save('./model_{}.hd5'.format(episode))

    writer.add_scalar("Reward/Train", score, episode)
    writer.add_scalar("Duration/Train", step_counter, episode)

    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    writer.add_scalar("Epsilon", eps_threshold, episode)

    steps_done += 1


env.close()
policy_net.save('./model_latest.hd5')
