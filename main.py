import os
import gym
import numpy as np
import torch
from torch.autograd import Variable

import buffer
import model
import train

MAX_EPISODES = 5000
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300

env = gym.make("LunarLander-v2", render_mode="human")
S_DIM = np.array(env.observation_space.sample()).size
A_DIM = np.array(env.action_space.sample()).size
A_DIS = env.action_space.n

print("State Dimensions: ", S_DIM)
print("Action Dimensions: ", A_DIM)
print("Acton Distributions: ", A_DIS)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_DIS, ram)


for ep in range(MAX_EPISODES):
    observation, info = env.reset()
    print("EPISODE: ", ep, end="")

    total = 0
    for t in range(MAX_STEPS):
        s1 = np.float32(observation)

        a1 = trainer.selectAction(s1)

        s2, r1, terminated, truncted, info = env.step(a1)

        total += r1
        if terminated:
            s2 = None
            print(" LENGTH: ", t, " REWARD: ", total)
        else:
            s2 = np.float32(s2)
            ram.add(s1, a1, r1, s2)

        observation = s2

        trainer.optimize()
        if terminated: break

    if ep % 100 == 0: trainer.save(ep)


print("Episodes Completed!!!")
