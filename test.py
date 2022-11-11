from model import ActorCritic
import torch
import gym
import gobang
from const import *



def test(n_episodes=5, name=None):
    env = gym.make('lazyGO-v0', render_mode='human')

    policy = ActorCritic().to(device)
    
    if name:
        policy.load_state_dict(torch.load('./trained/{}'.format(name)))
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()

        running_reward = 0
        for t in range(10000):
            state = torch.from_numpy(state).float().to(device)
            action = policy(state)
            state, reward, done, _, _ = env.step(action)
            running_reward += reward
            if done: break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))

    env.close()
            



if __name__ == '__main__':
    test()
