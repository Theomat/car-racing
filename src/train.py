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

from copy import deepcopy

EPISODES = 5000
RENDER = False
SKIP_ZOOM = True
FRAME_STACK = 3
BATCH_SIZE = 256
GAMMA = 0.96
LR = 0.001
MAX_NEG_STEPS = 80
SAVE_EVERY = 1000
UPDATE_EVERY = 128
EARLY_STOP = True
MEMORY_SIZE = 5000

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 0.002
CLOSE_EVERY = 5

env = gym.make('CarRacing-v0')

optimizer = Adam(lr=LR, epsilon=1e-7)
policy_net = DQN(FRAME_STACK, 84, ACTION_SPACE)
policy_net.compile(loss='mean_squared_error', optimizer=optimizer)


memory = ReplayMemory(MEMORY_SIZE)

writer = tf.summary.create_file_writer("./tf_logs")


steps_done = 0
t_step = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = max(EPS_END, EPS_START - EPS_DECAY * steps_done)

    if sample > eps_threshold:
        prediction = policy_net.predict(np.expand_dims(state, axis=0))[0]
        return np.argmax(prediction)
    else:
        return random.randrange(ACTION_SPACE)


def torchfy(obs):
    obs = np.float64(np.uint8(np.dot(obs[...,:3], [0.2989, 0.5870, 0.1140])))
    return deepcopy(obs[:84, 6:90]) / 255.0



def optimize_model(call):
    if len(memory) < BATCH_SIZE:
        print('hi')
        return
    transitions = memory.sample(BATCH_SIZE)
    states = np.stack([s for s, _, _, _ in transitions])
    next_states = np.stack([ns for _, _, ns, _ in transitions if ns is not None])


    next_state_values = np.argmax(policy_net.predict(next_states), axis=1)
    target_values = policy_net.predict(states)
    next_state_index = 0
    for i, transition in enumerate(transitions):
        s, a, ns, r = transition

        if ns is None:
            target_values[i, a] = r
        else:
            target_values[i, a] = r + GAMMA * next_state_values[next_state_index]
            next_state_index += 1

    history = policy_net.fit(states, target_values, epochs=1, batch_size=BATCH_SIZE, verbose=0)
    return np.mean(history.history['loss'])


opt_call = 0
for episode in tqdm(range(EPISODES)):

    done = False
    observation = env.reset()

    if SKIP_ZOOM:
        for _ in range(50):
            observation, _, _, _ = env.step(NOTHING)
            RENDER and env.render()

    state = np.zeros((84, 84, FRAME_STACK), dtype=np.float)

    for i in range(FRAME_STACK):
        state[:, :, i] = torchfy(observation)

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
            if score < 0 and step_counter > 300:
                done = True

        prev_state = deepcopy(state)

        for i in range(FRAME_STACK-1):
            state[:, :, i] = deepcopy(state[:, :, i + 1])
        state[:, :, -1] = torchfy(observation)

        memory.push(prev_state, action, state, reward)

        '''
        t_step = (t_step + 1) % UPDATE_EVERY
        if t_step == 0:
            opt_call += 1
        '''
        RENDER and env.render()
        step_counter += 1

    loss = optimize_model(opt_call)

    if episode % CLOSE_EVERY == 0:
        env.close()
        env = gym.make('CarRacing-v0')


    if episode % SAVE_EVERY == 0:
        policy_net.save('./model_{}.hd5'.format(episode))

    with writer.as_default():
        tf.summary.scalar("Reward/Train", score, step=episode)
        tf.summary.scalar("Duration/Train", step_counter, step=episode)
        tf.summary.scalar("Loss/Train", loss, step=episode)


        eps_threshold = max(EPS_END, EPS_START - EPS_DECAY * steps_done)
        tf.summary.scalar("Epsilon", eps_threshold, step=episode)

        print('score', score, episode)
        print('duration', step_counter, episode)
        print("eps", eps_threshold, episode)


    steps_done += 1

writer.close()
env.close()
policy_net.save('./model_latest.hd5')
