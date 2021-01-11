import torch.nn as nn
import torch.nn.functional as F

CONVOLUTIONAL_FILTERS = 32


class ResidualLayer(nn.Module):
    def __init__(self):
        super(ResidualLayer, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(CONVOLUTIONAL_FILTERS, CONVOLUTIONAL_FILTERS, 3),
            nn.BatchNorm2d(CONVOLUTIONAL_FILTERS),
            nn.ReLU(),
            nn.Conv2d(CONVOLUTIONAL_FILTERS, CONVOLUTIONAL_FILTERS, 3),
            nn.BatchNorm2d(CONVOLUTIONAL_FILTERS),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, blocks=9):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList([ResidualLayer() for _ in range(blocks)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        return x


class PolicyModel(nn.Module):
    def __init__(self, action_space: int):
        super(PolicyModel, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(12, 32, 3),
            nn.ReLU()
        )
        blocks: int = 9
        self.residual_block = ResidualBlock(blocks)
        size: int = 96 - (1 + blocks) * 4 + 2
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * size * size, action_space),
            nn.ReLU(),
            nn.Linear(action_space, action_space)
        )

    def forward(self, x):
        y = self.initial(x)
        y = self.residual_block(y)
        return self.layers(y)
