import numpy as np
import gym
import gobang
import model
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using {}'.format(device))



env = gym.make('lazyGO-v0', render_mode="human")
sDim = np.array(env.observation_space.sample()).size
aDim = env.action_space.n


ac = model.ActorCritic(sDim, aDim, device)


ac.load('./trained/actor/actor.pth', './trained/critic/critic.pth')


for _ in range(10):
    state, info = env.reset()

    done = False
    while not done:
        action = ac.policy(state, state)

        state, reward, done, trunk, info = env.step(action)

        if info:
            print(info)
