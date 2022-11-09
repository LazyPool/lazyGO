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
            action = self.select(policy(state1))
            reward, terminal = self.playground.perform(action)
            state2 = self.playground.getState()
            self.buffer.update(state1, action, reward, state2)
        self.buffer.store()

    def select(self, guide):
        index = list(range(len(guide)))
        possible = random.choices(index, weights=guide, k=1)[0]

        dim0 = possible // 64
        dim1 = possible % 64 // 8
        dim2 = possible % 64 % 8

        return torch.Tensor([dim0, dim1, dim2]).type(torch.long)

class Buffer:
    def __init__(self):
        self.capsity = 256
        self.storage = []
        with open('Buffer', 'r') as f:
            for line in f.readlines():
                splitedLine = line.replace('\n','').split(';')
                state1 = splitedLine[0]
                action = splitedLine[1]
                reward = splitedLine[2]
                state2 = splitedLine[3]

                state1 = torch.Tensor([float(ch) for ch in list(state1)])
                action = torch.Tensor([float(ch) for ch in list(action)])
                reward = float(reward)
                state2 = torch.Tensor([float(ch) for ch in list(state2)])

                self.storage.append((state1, action, reward, state2))

    def update(self, state1, action, reward, state2):
        self.storage.append((state1, action, reward, state2))
        if(len(self.storage)>self.capsity):
            self.storage.pop(0)

    def store(self):
        string = ""
        for record in self.storage:
            state1 = record[0]
            action = record[1]
            reward = record[2]
            state2 = record[3]

            state1 = ''.join(str(int(v)) for v in state1)
            action = ''.join(str(int(v)) for v in action)
            reward = str(reward)
            state2 = ''.join(str(int(v)) for v in state2)

            string += ';'.join([state1, action, reward, state2])
            string += '\n'
        with open('Buffer', 'w') as f:
            f.write(string)

    def clear(self):
        string = ""
        with open('Buffer', 'w') as f:
            f.write(string)
