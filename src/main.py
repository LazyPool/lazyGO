from Env import Environment
from Actor import Actor
from Critic import Critic
from Hypers import *
import time



def main():
    env = Environment()

    actor = Actor(env)
    critic = Critic(env)

    for episode in range(EPISODE):
        state1 = env.reset()

        for step in range(STEP):
            action = actor.selectA(state1)

            state2, reward, terminal = env.step(action)

            TD_error = critic.trainQnet(state1, reward, state2)

            actor.learn(state1, action, TD_error)

            state1 = state2
            if terminal: break

        if episode % C == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    action = actor.selectA(state)
                    
                    state, reward, terminal = env.step(action)

                    total_reward += reward
                    if terminal: break

            aveRward = total_reward/TEST
            print("episode:", episode, "Evaluation Average Reward:", aveRward)




if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('Total time is', time_end-time_start,'s')
