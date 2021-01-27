import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, input_channels, input_size, output_size):

        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of the image when squashed to a linear vector
        # We assume here that the input image is square
        conv_length = self._conv_shape(
            self._conv_shape(self._conv_shape(input_size, 4, 4), 4, 2), 3, 1
        )
        conv_shape = conv_length ** 2 * 64
        self.linear1 = nn.Linear(conv_shape, 512)
        self.linear2 = nn.Linear(512, output_size)

    @staticmethod
    def _conv_shape(input_size, filter_size, stride, padding=0):
        return 1 + (input_size - filter_size + 2 * padding) // stride

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear1(x.view(x.size(0), -1)))
        return self.linear2(x)
