import torch
import torch.nn as nn
import torch.nn.functional as F

class TicTacToeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(27, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.policy_head = nn.Linear(64, 9)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(-1, 27)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value
