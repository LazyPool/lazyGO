import torch
import torch.autograd as Variable

import numpy as np


def softUpdate(source, target, tau):
    for param1, param2 in zip(source.parameters(), target.parameters()):
        param2.data.copy_(
            param2.data * (1. - tau) + param1.data * tau
        )

def hardUpdate(source, target):
    for param1, param2 in zip(source.parameters(), target.parameters()):
        param2.data.copy_(
            param1.data
        )


class OrnsteinUhlenbeckActionNoise:
	def __init__(self, aDim, mu = 0, theta = 0.15, sigma = 0.2):
		self.aDim = aDim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.aDim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X
