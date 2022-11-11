
# LazyGO - A very lazy AI trained by the LazyPool QAQ



## Abstract
Actor-Critic algorithm, which is oftenly refered as AC, has been proved to be a good algorithm in the Reinforcement Learning(RL) in many domains. In the AC, the Actor estimates the probability of an action $a$ under a given state $s$ and choose the $a$ somehow randomly according to a policy function $\pi$. In constrant to Actor, Critic is a function that predict the expected return $R$ of the whole game when performing the action $a$ under the state $s$. The Actor use the Policy-Gradient as its optimization, while Critic may will often be binned with DQN.

Actor/Policy-Gradient:[(more details)](https://github.com/LazyPool/lazyGO/blob/main/doc/Actor.md)
$$J(\theta)\approx\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^TR({\tau}^n)\nabla_{\theta}log{P(a_{t}^{n}|s_{t}^{n};\theta)}$$

Critic/DQN:[(more details)](https://github.com/LazyPool/lazyGO/blob/main/doc/Critic.md)
$$Q^\pi(s_{t},a_{t};\theta){\leftarrow}r_{t}+{\gamma}\underset{a}{\max}Q^\pi(s_{t+1},a_{t+1};\theta)$$

So the AC is exactly the combination of the Actor and Critic. In Actor, we use the $R({\tau}^n)$ to serve as the expected return of the whole game, but it's no very proper as it's just the simple sum of the rewards of $r_{t}$. So we may try to use the $Q^\pi(s_{t},a_{t};\omega)$ to substitude the simple $R$, which could estimate the expected return more proper. Constractly in Critic, we simply use $\pi={\arg\max_{a}}Q^\pi(s_{t},a_{t};\omega)$ as the universe policy so that updating the Q-net leads to updating the agent's policy. But always select the max would not always lead to the good result. So now we use the function simulated by the policy-net as the policy $\pi$ so that we could train a better critic.

Actor-Critic:[(more details)](https://github.com/LazyPool/lazyGO/blob/main/doc/Actor-Critic.md)
$$J(\theta)\approx\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^TQ^\pi(s_{t}^{n},a_{t}^{n};\omega)\nabla_{\theta}log{P(a_{t}^{n}|s_{t}^{n};\theta)}$$

In this program, I will try to train a simple AI based on the Actor-Critic framework. The AI is designed to play a very easy game, which is the Gobang with a only 8x8 board. The agent will try to interact with the board and train 2 network to learn how to win the Gobang. The network is designed to build using pytorch package, there some particulars such as how to make the agent be able to interact with the envrionment and how the envirionment update its state etc.

In "Introduction", the program's components will be present and elaborated carefully, where show the conception about this program and the links among the components so that could guide the clearer method disign. In "Method", I would like to show my details design about every component and even show some the pseudo-code. In "Result", there would be some expriments for the trained AI so that we can estimate its talence. At last, in "Discussion", we may discuss some problems and try to deeply analyse this program's advantage and disadvantages.



## Introduction



## Method



## Result



## Discussion
