import os
import torch
import random
from Environment import Environment
from Actor import Actor

class Agent:
    def __init__(self, env, policy, buffer):
        self.playground = env
        self.policy = policy
        self.buffer = buffer

    def play(self):
        terminal = False
        while(not terminal):
            state1 = self.playground.getState()
            action = self.select(self.policy(state1))
            reward, terminal = self.playground.perform(action)
            state2 = self.playground.getState()
            self.buffer.update(state1, action, reward, state2)
        self.playground.clear()

    def select(self, guide):
        possible = torch.multinomial(guide, 1)

        dim0 = possible // 64
        dim1 = possible % 64 // 8
        dim2 = possible % 64 % 8

        return torch.Tensor([dim0, dim1, dim2]).type(torch.long)

class Buffer:
    def __init__(self):
        self.capsity = 1024
        self.storage = []

    def update(self, state1, action, reward, state2):
        self.storage.append((state1, action, reward, state2))
        while(len(self.storage)>self.capsity):
            self.storage.pop(0)

    def getSamples(self, N):
        return [random.choice(self.storage) for _ in range(N)]
