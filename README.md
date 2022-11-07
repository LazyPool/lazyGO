
# LazyGO - A very lazy AI trained by the LazyPool QAQ

## Abstract
Actor-Critic algorithm, which is oftenly reffered as AC, has been proved to be a good algorithm in the Reinforcement Learning(RL) in many domains. In the AC, the Actor is a function that estimate the probability of an action $a$ under a given state $s$ and choose the $a$ somehow randomly according to a policy function $\pi$. In constrant to Actor, Critic is a function that predict the expected return $R$ of the whole game when performing the action $a$ under the state $s$. The Actor use the Policy-Gradient as its optimization, while Critic may will ofen be binned with DQN.

Actor/Policy-Gradient:[(more details)](https://github.com/LazyPool/lazyGO/blob/main/Actor.md)
$$J(\theta)\approx\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^NR({\tau}^n)\nabla_{\theta}log{P(a_{t}^{n}|s_{t}^{n};\theta)}$$

Critic/DQN:[(more details)](https://github.com/LazyPool/lazyGO/blob/main/Critic.md)
$$Q^\pi(s_{t},a_{t};\theta){\leftarrow}r_{t}+{\gamma}\underset{a}{\max}Q^\pi(s_{t+1},a_{t+1};\theta)$$

So the AC is exactly the combination of the Actor and Critic, hence usually needs to train 2 neural networks, one of which is used to simulate the Actor, while the other one of which is used to simulate the Critic. So the key problem is how to make the two interact with each other and finally converge to a nice balance point, where the agent perform as expected.

In this program, I will try to train a simple AI based on the Actor-Critic framework.

## Introduction

## Method

## Result

## Discussion
