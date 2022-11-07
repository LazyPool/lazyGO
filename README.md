
# LazyGO - A very lazy AI trained by the LazyPool QAQ

## Abstract
Actor-Critic algorithm, which is oftenly reffered as AC, has been proved to be a good algorithm in the Reinforcement Learning(RL) in many domains. In the AC, the Actor is a function that estimate the probability of an action $a$ under a given state $s$ and choose the according to a policy function $\pi$. In constrant to Actor, Critic is a function that predict the expected return $R$ of the whole game while performing the action $a$ under the state $s$. The Actor will ofen be binned with the Policy-Gradient, while Critic may use DQN as its optimization.

Actor/Policy-Gradient:[(more details)](./Actor.md)
$$\nabla_{\theta}\widehat{R(\tau)}\approx\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^NR({\tau}^n)\nabla_{\theta}log{P(a_{t}^{n}|s_{t}^{n};\theta)}$$

Critic/DQN:[(more details)](./Critic.md)
$$Q^\pi(s_{t},a_{t};\theta){\leftarrow}r_{t}+{\gamma}Q^\pi(s_{t+1},a_{t+1};\theta)$$

## Introduction

## Method

## Result

## Discussion
