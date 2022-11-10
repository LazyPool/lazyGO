import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from Hypers import *



class PGNetwork(nn.Module):
    def __init__(self, sDim, aDim):
        super(PGNetwork, self).__init__()
        self.fc1 = nn.Linear(sDim, 128)
        self.fc2 = nn.Linear(128, aDim)



    def forward(self, x):
        out = F.relu(self.fc1(x))
        aScore = self.fc2(out)
        aProb = F.softmax(aScore, dim=0)
        return aProb



class Actor(object):
    def __init__(self, env):
        self.sDim = env.sDim
        self.aDim = env.aDim

        self.network = PGNetwork(self.sDim, self.aDim)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.lFN = torch.nn.NLLLoss()



    def selectA(self, state):
        state = state.reshape(-1)
        aProb = self.network(state)

        self.m = Categorical(aProb)
        a = self.m.sample()

        return a.item()


    def learn(self, state, action, TD_error):
        state = state.reshape(-1)
        aProb = self.network(state)

        action = torch.LongTensor([action])

        negLog = self.lFN(torch.log(aProb).unsqueeze(0), action)

        loss = negLog * TD_error

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
