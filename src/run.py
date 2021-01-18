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
MAX_STEPS = 1000
MEMORY_BUFFER_SIZE = 10**5
EPISODES = 1000
TEST_EPISODES = 0
TRAIN_FREQUENCY = 10
BATCH_SIZE = 32
EPS_START = .7
LR = 0.001
L2_REG_COEFF = 1e-4
GAMMA = .98
K = .1
# Instanciation ===============================================================
model = PolicyModel(discretize.MAX_ACTION).float().to(device)
replay_buffer = UniformReplayBuffer(MEMORY_BUFFER_SIZE)
epsilon = annealing.linear(EPS_START, 0, EPISODES)
policy = policies.from_model(model)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2_REG_COEFF)


# Functions====================================================================
def loss_function(states, actions, rewards, next_states):
    q_values = model(states)
    action_values = torch.gather(q_values, 1, actions.to(torch.int64))
    next_states_values, _ = model(next_states).max(1)
    expected_values = rewards + GAMMA * next_states_values.view((next_states_values.shape[0], 1))
    # DQN reg loss
    return F.mse_loss(action_values, expected_values) + torch.mean(K * q_values)


def optimize_model(writer, training_step):
    total_loss = 0.0
    for _ in range(TRAIN_FREQUENCY * EPISODES // BATCH_SIZE):
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

        total_loss += loss.item()

    writer.add_scalar('Loss/Train', total_loss, training_step)
    # Set back to eval mode
    model.eval()


# Run =========================================================================
print(model)
trainer.train(policy, optimize_model, epsilon, replay_buffer,
              EPISODES, TRAIN_FREQUENCY, TEST_EPISODES, MAX_STEPS)
