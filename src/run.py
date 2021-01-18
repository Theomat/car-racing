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
device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
device_name: str = "cpu" if device == "cpu" else torch.cuda.get_device_name(0)
print('Training on ' + device_name)
# Globals =====================================================================
MAX_STEPS = 1000
FRAMES_STACK = 3
TRAIN_FREQUENCY = 2
MEMORY_BUFFER_SIZE = MAX_STEPS * TRAIN_FREQUENCY
EPISODES = 1000
TEST_EPISODES = 0
BATCH_SIZE = 32
EPS_START = .7
LR = 1e-4
L2_REG_COEFF = 1e-3
GAMMA = .98
K = .1
TARGET_MODEL_UPDATE_FREQUENCY = 5
MAX_CONSECUTIVE_NEGATIVE_STEPS = 25
# Instanciation ===============================================================
model: torch.nn.Module = PolicyModel(discretize.MAX_ACTION, FRAMES_STACK).float().to(device)
target_model: torch.nn.Module = PolicyModel(discretize.MAX_ACTION, FRAMES_STACK).float().to(device)
replay_buffer: UniformReplayBuffer = UniformReplayBuffer(MEMORY_BUFFER_SIZE, FRAMES_STACK)
epsilon: annealing.Annealing = annealing.linear(EPS_START, 0, EPISODES)
policy: policies.Policy = policies.from_model(model)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2_REG_COEFF)


# Functions====================================================================
def loss_function(states: torch.FloatTensor, actions: torch.LongTensor,
                  rewards: torch.FloatTensor, next_states: torch.FloatTensor) -> torch.FloatTensor:
    q_values: torch.FloatTensor = model(states)
    action_values: torch.FloatTensor = torch.gather(q_values, 1, actions)
    next_states_values, _ = target_model(next_states).max(1)
    expected_values: torch.FloatTensor = rewards + GAMMA * next_states_values.view((next_states_values.shape[0], 1))
    # DQN reg loss
    return F.mse_loss(action_values, expected_values)  # + torch.mean(K * q_values)


def optimize_model(writer, training_step: int):
    total_loss: float = 0.0
    for _ in range(TRAIN_FREQUENCY * MAX_STEPS // BATCH_SIZE):
        states, actions, rewards, next_states = replay_buffer.sample(BATCH_SIZE)
        # Convert to correct type tensors
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)

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

    if training_step % (TRAIN_FREQUENCY * TARGET_MODEL_UPDATE_FREQUENCY) == 0:
        target_model.load_state_dict(model.state_dict())


# Run =========================================================================
print(model)
trainer.train(policy, optimize_model, epsilon, replay_buffer,
              EPISODES, TRAIN_FREQUENCY,
              frames_stack=FRAMES_STACK,
              test_performance_episodes=TEST_EPISODES,
              max_steps=MAX_STEPS,
              max_negative_steps=MAX_CONSECUTIVE_NEGATIVE_STEPS)
