import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from const import *


class ActorCritic(nn.Module):
    def __init__(self, sDim, aDim):
        super(ActorCritic, self).__init__()
        self.flatten = nn.Flatten(0)
        self.affline = nn.Linear(sDim, 128)

        self.action_layer = nn.Linear(128, aDim)
        self.value_layer = nn.Linear(128, 1)

        self.logprobs = []
        self.state_values = []
        self.rewards = []


    def forward(self, state):
        state = self.flatten(state)
        state = F.relu(self.affline(state))

        state_value = self.value_layer(state)
        
        action_probs = F.softmax(self.action_layer(state), dim=0)
        action_distribution = Categorical(action_probs)

        action = action_distribution.sample()

        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)

        print(action)
        return action.item()


    def calculateLoss(self, gamma=0.99):
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)

        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std())

        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward - value.item()
            action_loss = - logprob * advantage

            reward = reward.reshape(value.shape)
            value_loss = F.smooth_l1_loss(value, reward)

            loss += (action_loss + value_loss)
        return loss


    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
