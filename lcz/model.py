import torch
import torch.nn as nn
import torch.nn.functional as F
from config import IN_CHANNELS, NUM_RES_BLOCKS, NUM_FILTERS, ACTION_SIZE

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class LeelaChessZeroNet(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, num_res_blocks=NUM_RES_BLOCKS,
                 num_filters=NUM_FILTERS, action_size=ACTION_SIZE):
        super().__init__()
        self.in_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.action_size = action_size

        self.conv_init = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(num_filters)
        self.dropout = nn.Dropout(p=0.2)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # policy head
        self.conv_policy = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * 8 * 8, action_size)

        # value head
        self.conv_value = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(1 * 8 * 8, num_filters)
        self.fc_value2 = nn.Linear(num_filters, 1)

    def forward(self, x):
        # handle empty batch
        batch_size = x.size(0)
        if batch_size == 0:
            # return empty policy and value tensors
            return x.new_empty((0, self.action_size)), x.new_empty((0, 1))
        x = F.relu(self.bn_init(self.conv_init(x)))
        x = self.dropout(x)
        for block in self.res_blocks:
            x = block(x)
        # policy head
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.reshape(x.size(0), -1)
        p = F.log_softmax(self.fc_policy(p), dim=1)
        # value head
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.reshape(x.size(0), -1)
        v = F.relu(self.fc_value1(v))
        v = torch.tanh(self.fc_value2(v))
        return p, v 
