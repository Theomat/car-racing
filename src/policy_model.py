import torch.nn as nn


class PolicyModel(nn.Module):
    def __init__(self, action_space: int, frames_stack: int):
        super(PolicyModel, self).__init__()

        self.layers = nn.Sequential(
            # nn.Conv2d(frames_stack, 6, 7),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.BatchNorm2d(6),
            # nn.Conv2d(6, 12, 4),
            # nn.MaxPool2d(2),
            # nn.BatchNorm2d(12),
            # nn.Flatten(1),
            # nn.Linear(5292, 216),
            # nn.ReLU(),
            # nn.Linear(216, action_space)
            nn.Flatten(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
            # nn.Linear(216, action_space)
        )

    def forward(self, x):
        return self.layers(x)
