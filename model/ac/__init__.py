from model.ac.network import Actor
from model.ac.network import Critic
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn.functional as F
import numpy as np



gamma = 0.99
lr = 1e-3
betas = (0.9, 0.99)



class ActorCritic(object):
    def __init__(self, sDim, aDim, device='cpu'):
        self.device = device

        self.actor = Actor(sDim, aDim).to(self.device)
        self.critic = Critic(sDim, aDim).to(self.device)

        self.states = []
        self.noactions = []
        self.actions = []
        self.rewards = []

        self.optActor = optim.Adam(self.actor.parameters(), lr=lr, betas=betas)
        self.optCritic = optim.Adam(self.critic.parameters(), lr=lr, betas=betas)


    def policy(self, state, noaction):
        state = torch.from_numpy(state).type(torch.float)
        noaction = torch.from_numpy(noaction).type(torch.float)

        self.states.append(np.array(state))
        self.noactions.append(np.array(noaction))

        state = torch.unsqueeze(state, dim=0).to(self.device)
        noaction = torch.unsqueeze(noaction, dim=0).to(self.device)

        actionProb = self.actor(state, noaction)
        
        self.actions.append(np.array(actionProb[0].detach().to('cpu')))

        actionDis = Categorical(actionProb)

        action = actionDis.sample()

        return action.item()


    def obtain(self, reward):
        for i in range(len(self.rewards)):
            dis = 1
            for _ in range(len(self.rewards) - i):
                dis *= gamma
            self.rewards[i] += reward * dis
        self.rewards.append(reward)


    def learn(self, fin):
        states = torch.tensor(np.array(self.states)).to(self.device)
        noactions = torch.tensor(np.array(self.noactions)).to(self.device)
        actions = torch.tensor(np.array(self.actions)).to(self.device)
        rewards = torch.tensor(self.rewards).unsqueeze(1).to(self.device)

        values = self.critic(states, actions)

        self.optCritic.zero_grad()
        loss = F.smooth_l1_loss(values, rewards)
        loss.backward()
        self.optCritic.step()

        actionProbs = self.actor(states, noactions)
        self.optActor.zero_grad()
        loss = -1 * torch.sum(self.critic(states, actionProbs))
        loss.backward()
        self.optActor.step()


    def forget(self):
        del self.states[:]
        del self.noactions[:]
        del self.actions[:]
        del self.rewards[:]


    def save(self, actorPath, criticPath):
        torch.save(self.actor.state_dict(), actorPath)
        torch.save(self.critic.state_dict(), criticPath)


    def load(self, actorPath, criticPath):
        self.actor.load_state_dict(torch.load(actorPath))
        self.critic.load_state_dict(torch.load(criticPath))
