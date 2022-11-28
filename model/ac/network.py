import torch
import torch.nn as nn
import torch.nn.functional as F



class Actor(nn.Module):
    def __init__(self, sDim, aDim):
        super(Actor, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(sDim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, aDim)


    def forward(self, state, noaction):
        state = self.flatten(state)
        noaction = self.flatten(noaction)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = self.fc4(x)

        action -= noaction * 999999999.

        return F.softmax(action, dim=1)



class Critic(nn.Module):
    def __init__(self, sDim, aDim):
        super(Critic, self).__init__()

        self.flatten = nn.Flatten()

        self.fcs1 = nn.Linear(sDim, 128)
        self.fca1 = nn.Linear(aDim, 128)

        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)


    def forward(self, state, action):
        state = self.flatten(state)
        action = self.flatten(action)

        s1 = F.relu(self.fcs1(state))
        a1 = F.relu(self.fca1(action))
        x = torch.cat((s1, a1), dim=1)

        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
