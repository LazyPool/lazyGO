import sys
import torch
import random
from Environment import Environment
from Agent import Agent, Buffer
from Actor import Actor
from Critic import Critic

def train(actor, critic, buffer):
    N = 512
    alpha = 1e-10
    beta = 1e-10
    gamma = 0.618

    critic_dict = critic.state_dict()
    actor_dict = actor.state_dict()
    
    samples = buffer.getSamples(N)
    for record in samples:
        s1 = record[0]
        a1 = record[1]
        r1 = record[2]
        s2 = record[3]

        guide = actor(s2)
        index = torch.multinomial(guide, 1)
        a2 = index2action(index)
        
        s1_a1 = torch.cat([s1, a1])
        s2_a2 = torch.cat([s2, a2])
        q1 = critic(s1_a1)
        q2 = critic(s2_a2)

        delta = q1 - (r1 + gamma * q2)

        d_omega = critic(s1_a1)
        d_omega.backward()
        critic_iteral = critic.named_parameters()
        for name, param in critic_iteral:
            param_new = param - beta * delta * param.grad
            critic_dict[name] = param_new

        index = action2index(a1).type(torch.long)
        target = actor(s1)[index]
        d_theta = torch.log(target+1e-8)
        d_theta.backward()
        actor_iteral = actor.named_parameters()
        for name, param in actor_iteral:
            param_new = param + alpha * delta * param.grad
            actor_dict[name] = param_new 
        
        actor.load_state_dict(actor_dict)
        critic.load_state_dict(critic_dict)

    torch.save(actor.state_dict(), 'actor.pth')
    torch.save(critic.state_dict(), 'critic.pth')

def index2action(index):
    dim0 = index // 64
    dim1 = index % 64 // 8
    dim2 = index % 64 % 8
    return torch.Tensor([dim0, dim1, dim2]).type(torch.long)

def action2index(action):
    dim0 = action[0]
    dim1 = action[1]
    dim2 = action[2]
    return dim0 * 64 + dim1 * 8 + dim2

if __name__ == '__main__':
    env = Environment()

    actor = Actor()
    critic = Critic()

    buffer = Buffer()
    ai = Agent(env, actor, buffer)

    tips = input("Train a totally new AI?y/[N]")
    if tips == 'n' or tips == 'N' or tips == '':
        print("loading the trained parameters")
        actor.load_state_dict(torch.load("actor.pth"))
        critic.load_state_dict(torch.load("critic.pth"))

    tips = input("Want to update the Buffer?[Y]/n")
    if tips == 'y' or tips == 'Y' or tips == '':
        times = 256
        for i in range(times):
            prog = i * 100 // times
            ai.play()
            print("\r", end="")
            print("Update: {}%: ".format(prog), "▋" * (prog // 2), end="")
            sys.stdout.flush()
    print('\n')

    numbers = input("How many times would you like to train?")
    numbers = int(numbers)
    for i in range(numbers):
        prog = i * 100 // numbers 
        train(actor, critic, buffer)
        print("\r", end="")
        print("Update: {}%: ".format(prog), "▋" * (prog // 2), end="")
        sys.stdout.flush()
    print('\n')
