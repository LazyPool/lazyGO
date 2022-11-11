from model import ActorCritic
import torch
import torch.optim as optim
import gym
import gobang
from const import *
import sys
import numpy as np



def train():
    torch.manual_seed(random_seed)
    
    env = gym.make("lazyGO-v0", render_mode="human")
    sDim = np.array(env.observation_space.sample()).size
    aDim = env.action_space.n
    
    policy = ActorCritic(sDim, aDim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)
    
    running_reward = 0
    for episode in range(EPISODE):
        print("\r", end="")
        print("Progress: {}%: ".format(episode * 100 // 3000), "▋" * (episode * 100 // 3000), end="")
        sys.stdout.flush()
            
        state, _ = env.reset()
        for t in range(STEPS):
            state = torch.from_numpy(state).float()
            state = state.to(device)

            action = policy(state)
            state, reward, done, _,  _ = env.step(action)

            reward = torch.tensor(reward)
            policy.rewards.append(reward)
            running_reward += reward
            
            if done: break
                    
        optimizer.zero_grad()
        loss = policy.calculateLoss(gamma)
        loss.backward()
        optimizer.step()        
        policy.clearMemory()

        if running_reward > 4000: print("the train stop advancedly!"); break
        if episode % 20 == 0: running_reward = 0
        
    torch.save(policy.state_dict(), "./trained/lazyGO_{}_{}_{}.pth".format(lr, betas[0], betas[1]))
    print("====================saved successfully!====================")




if __name__ == '__main__':
    print("Using {} device.".format(device))
    train()
