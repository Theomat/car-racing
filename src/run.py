from policy_model import PolicyModel
from uniform_replay_buffer import UniformReplayBuffer
import discretize
import annealing
import trainer
import policies

from typing import List

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
TRAIN_FREQUENCY = 1
MEMORY_BUFFER_SIZE = MAX_STEPS * TRAIN_FREQUENCY * 5
EPISODES = 5000
TEST_EPISODES = 0
BATCH_SIZE = 32
BATCH_PER_TRAINING_STEP = 128 // BATCH_SIZE
EPS_START = 1
LR = 1e-3
L2_REG_COEFF = 1e-7
GAMMA = .95
K = 0
TARGET_MODEL_UPDATE_FREQUENCY = 2
MAX_CONSECUTIVE_NEGATIVE_STEPS = 25
# Instanciation ===============================================================
model: torch.nn.Module = PolicyModel(discretize.MAX_ACTION, FRAMES_STACK).float().to(device)
target_model: torch.nn.Module = PolicyModel(discretize.MAX_ACTION, FRAMES_STACK).float().to(device)
replay_buffer: UniformReplayBuffer = UniformReplayBuffer(MEMORY_BUFFER_SIZE, FRAMES_STACK)
epsilon: annealing.Annealing = annealing.linear(EPS_START, 0.05, 2000)
policy: policies.Policy = policies.from_model(model)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2_REG_COEFF)


# Init model
model.eval()
# Init target net
target_model.load_state_dict(model.state_dict())
target_model.eval()


# Functions====================================================================
def loss_function(states: torch.FloatTensor, actions: torch.LongTensor,
                  rewards: torch.FloatTensor, next_states: List[torch.FloatTensor],
                  writer, step: int) -> torch.FloatTensor:

    q_values: torch.FloatTensor = model(states)
    action_values: torch.FloatTensor = torch.gather(q_values, 1, actions)

    non_final_mask: torch.BoolTensor = torch.tensor(tuple(map(lambda s: s is not None, next_states)),
                                                    device=device, dtype=torch.bool)
    non_final_next_states: torch.FloatTensor = torch.cat([s for s in next_states if s is not None])

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_model(non_final_next_states.view((-1, FRAMES_STACK, 96, 96))).max(1)[0].detach()
    expected_values: torch.FloatTensor = rewards + GAMMA * next_state_values.view((rewards.shape[0], 1))

    if writer is not None:
        residual_variance: float = (torch.var(expected_values - action_values) / torch.var(action_values)).item()
        writer.add_scalar('Residual Variance', residual_variance, step)
        writer.add_histogram('Action Values/Predicted', action_values, step)
        writer.add_histogram('Action Values/Target', expected_values, step)
        writer.add_histogram('Action Values/Error', expected_values - action_values, step)
    # DQN reg loss
    return F.mse_loss(action_values, expected_values) + K * torch.mean(action_values)


def optimize_model(writer, training_step: int):
    total_loss: float = 0.0
    # Set to train mode
    model.train()
    for i in range(BATCH_PER_TRAINING_STEP):
        states, actions, rewards, next_states = replay_buffer.sample(BATCH_SIZE)
        # Convert to correct type tensors
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)

        loss = loss_function(states, actions, rewards, next_states, writer, BATCH_PER_TRAINING_STEP * training_step + i)
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
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
              max_negative_steps=MAX_CONSECUTIVE_NEGATIVE_STEPS,
              renderTrain=False,
              renderTest=True)
