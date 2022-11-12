import gym
import gobang
import ac


env = gym.make("LunarLander-v2", render_mode="human")

trainer = ac.ACTrainer(env)

trainer.train(10000)
