import torch
import torch.nn as nn
import torch.nn.functional as F
from Hypers import *



class QNetwork(nn.Module):
    def __init__(self, sDim, aDim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(sDim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        res = self.fc2(out)
        return res



class Critic(object):
    def __init__(self, env):
        self.sDim = env.sDim
        self.aDim = env.aDim

        self.network = QNetwork(self.sDim, self.aDim)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.lFN = nn.MSELoss()


    def trainQnet(self, state1, reward, state2):
        s1, s2 = state1.reshape(-1), state2.reshape(-1)

        v1 = self.network(s1)
        v2 = self.network(s2).detach()

        loss = self.lFN(reward + GAMMA * v2, v1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            TD_error = reward + GAMMA * v2 - v1

        return TD_error
