
# LazyGO - A very lazy ai trained by the lazypool QAQ

## Abstract
Actor-Critic algorithm, which is oftenly reffered as AC, has been proved to be a good algorithm in the Reinforcement Learning(RL) in many domains. In the AC, the Actor is a function that estimate the probability of an action $a$ under a given state $s$ and choose the according to a policy function $\pi$. In constrant to Actor, Critic is a function that predict the expected return $R$ of the whole game while performing the action $a$ under the state $s$. The Actor will ofen be binned with the Policy-Gradient, while Critic may use DQN as its optimization.

$$\nabla_{\theta}\widehat{U(\tau)}\approx\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^NU({\tau}^n)\nabla_{\theta}log{P(a_{t}^{n}|s_{t}^{n};\theta)}$$

$$$$

## Introduction

## Method

## Result

## Discussion
