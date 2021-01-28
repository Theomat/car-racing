import gym
import numpy as np
import random
import math

from dqn import DQN
from replay_memory import ReplayMemory
from actions import discrete2cont, NOTHING, ACTION_SPACE

import tensorflow as tf
from tensorflow.keras.optimizers import Adam


from tqdm import tqdm


EPISODES = 5000
RENDER = False
SKIP_ZOOM = True
FRAME_STACK = 4
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

obs = env.reset()


for k in range(50):
    obs, _, _, _ = env.step([0,0,0])

optimizer = Adam(lr=LR, epsilon=1e-7)
policy_net = DQN(FRAME_STACK, 84, ACTION_SPACE)
policy_net.compile(loss='mean_squared_error', optimizer=optimizer)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

obs = np.float64(np.uint8(np.dot(obs[...,:3], [0.2989, 0.5870, 0.1140])))

imgplot = plt.imshow(obs[:84, 6:90]/255., cmap='gray')
plt.show()

