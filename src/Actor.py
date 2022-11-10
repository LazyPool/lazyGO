import torch
import torch.nn as nn
import torch.nn.functional as F



class PGNetwork(nn.Module):
    def __init__(self, sDim, aDim):
        super(PGNetwork, self).__init__()
        self.fc1 = nn.Linear(sDim, 20)
        self.fc2 = nn.Linear(20, aDim)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        aScore = self.fc2(x)
        aProb = F.softmax(aProb, dim=1)
        return res



class Actor(object):
    def __init__(self, env):
        self.sDim = 2*8*8
        self.aDim = 2*8*8

        self.network = PGNetwork(self.sDim, self.aDim)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.lFN = torch.nn.NLLLoss()


    def selectA(self, state):
        state = state.unsqueeze(0)
        aProb = self.network(state)

        m = Categorical(aProb)
        a = m.sample()

        return a.item()


    def learn(self, state, action, TD_error):
        state = state.unsqueeze(0)
        aProb = self.network(state)

        negLog = self.lFN(torch.log(aProb), action)

        loss = negLog * TD_error

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
