from policy_model import PolicyModel
from uniform_replay_buffer import UniformReplayBuffer
import discretize
import annealing
import trainer
import policies

import torch
import torch.optim as optim
import torch.nn.functional as F

# Setup =======================================================================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device_name = "cpu" if device == "cpu" else torch.cuda.get_device_name(0)
print('Training on ' + device_name)
# Globals =====================================================================
MEMORY_BUFFER_SIZE = 10**5
EPISODES = 1000
TEST_EPISODES = 10
TRAIN_FREQUENCY = 10
BATCH_SIZE = 128
EPS_START = .7
LR = 0.001
L2_REG_COEFF = 1e-4
GAMMA = .98
# Instanciation ===============================================================
model = PolicyModel(discretize.MAX_ACTION).float().to(device)
replay_buffer = UniformReplayBuffer(MEMORY_BUFFER_SIZE)
epsilon = annealing.linear(EPS_START, 0, EPISODES)
policy = policies.from_model(model)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2_REG_COEFF)


# Functions====================================================================
def loss_function(states, q_values, actions, rewards, next_states):
    q_values = model(states)
    action_values = torch.gather(q_values, 1, actions)
    next_states_values = model(next_states).max(1)
    expected_values = rewards + GAMMA * next_states_values
    return F.smooth_l1_loss(action_values, expected_values)


def optimize_model(writer, training_step):
    states, actions, rewards, next_states = replay_buffer.sample(BATCH_SIZE)
    # Convert to correct type tensors
    states = states.float().to(device)
    actions = actions.float().to(device)
    rewards = rewards.float().to(device)
    next_states = next_states.float().to(device)

    # Set to train mode
    model.train()

    loss = loss_function(states, actions, rewards, next_states)
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    writer.add_scalar('Loss/Train', loss.item(), training_step)
    # Set back to eval mode
    model.eval()


# Run =========================================================================
print(model)
trainer.train(policy, optimize_model, epsilon, replay_buffer,
              EPISODES, TRAIN_FREQUENCY, TEST_EPISODES)
