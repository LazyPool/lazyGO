
# LazyGO - A very lazy AI trained by the LazyPool QAQ



## Abstract
Actor-Critic algorithm, which is oftenly refered as AC, has been proved to be a good algorithm in the Reinforcement Learning(RL) in many domains. In the AC, the Actor cacculates the probability of each possible action $a$ under the given state $s$ by a policy function $\pi$ and then choose one roughly possible action according to a its probability distributions. Contrary to Actor, Critic is a function that predict the expected return $R$ of the whole game when performing the action $a$ under the state $s$. The Actor use the Policy-Gradient as its optimization, while Critic may will often be binned with DQN.

Actor/Policy-Gradient:[(more details)](https://github.com/LazyPool/lazyGO/blob/main/doc/Actor.md)
$$\nabla_{\theta}J(\theta)\approx\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^TR({\tau}^n)\nabla_{\theta}log{P(a_{t}^{n}|s_{t}^{n};\theta)}$$

Critic/DQN:[(more details)](https://github.com/LazyPool/lazyGO/blob/main/doc/Critic.md)
$$Q^\pi(s_{t},a_{t};\theta){\leftarrow}r_{t}+{\gamma}\underset{a}{\max}Q^\pi(s_{t+1},a_{t+1};\theta)$$

So the AC is exactly the combination of the Actor and Critic. In the Actor, we at first use the $R({\tau}^n)$ to serve as the expected return of the whole game, but it's no very proper as it's just the simple sum of the rewards of $r_{t}$. So we may try to use the $Q^\pi(s_{t},a_{t};\omega)$ to substitude the simple $R$, which could estimate the expected return more proper. Samely, in the Critic, we at first use $\pi={\arg\max_{a}}Q^\pi(s_{t},a_{t};\omega)$ as the universe policy so that updating the Q-net leads to updating the agent's policy. But always select the max would not always lead to the good result. So now we use the function simulated by the policy-net as the policy $\pi$ so that we could train a better critic.

Actor-Critic:[(more details)](https://github.com/LazyPool/lazyGO/blob/main/doc/Actor-Critic.md)
$$\nabla_{\theta}J(\theta)\approx\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^TQ^\pi(s_{t}^{n},a_{t}^{n};\omega)\nabla_{\theta}log{P(a_{t}^{n}|s_{t}^{n};\theta)}$$

In this program, I will try to train a simple AI based on the Actor-Critic framework. The AI is designed to play a very easy game, which is the Gobang with a only 8x8 board. The agent will try to interact with the board and train the Actor-Critic model to learn how to win the Gobang. The model is designed to be built using the pytorch package. More importantly, for convenience, the environment where the AI would interact with will be designend through the API supported by the OpenAI-gym's Env class. For more details about the OpenAI-gym, please refer to [OpenAI-gym](https://www.gymlibrary.dev/).



## Introduction
### Environment
#### Basic
The visual environment for the AI's interaction is based on the OpenAI-gym's Env class. Referring to the official documentation, I made a simple python module named "gobang" to be called during model training and testing. And here is the tree structure of this mod.

```
gobang
├── env
│   ├── __init__.py
│   └── lazygo.py
└── __init__.py

```

|Action Space|Discrete(4)|
|:-|:-|
|Observation Shape|(8,8)|
|Observation High|All elements are 2|
|Observatin Low|All elements are 0|

#### Discription
#### Actin Space
#### Observation Space
#### Rewards
#### Episode Termination


### Model


### Train



## Method



## Result



## Discussion
