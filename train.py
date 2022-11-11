from module import ActorCritic
import torch
import torch.optim as optim
import gym
from const import *
import sys



def train():
    gamma = 0.99
    lr = 0.02
    betas = (0.9, 0.999)
    
    random_seed = 543
    torch.manual_seed(random_seed)
    
    env = gym.make("LunarLander-v2", render_mode="human")
    
    policy = ActorCritic().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)
    
    running_reward = 0
    for i_episode in range(0, 3000):
        print("\r", end="")
        print("Progress: {}%: ".format(i_episode * 100 // 3000), "▋" * (i_episode * 10 // 3000), end="")
        sys.stdout.flush()
            
        state, _ = env.reset()
        for t in range(500):
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
        if i_episode % 20 == 0: running_reward = 0
        
    torch.save(policy.state_dict(), "./trained/LunarLand_{}_{}_{}.pth".format(lr, betas[0], betas[1]))
    print("====================saved successfully!====================")




if __name__ == '__main__':
    print("Using {} device.".format(device))
    train()
