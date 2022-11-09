
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
### Environment
Environment is the place where the Agent can observes the state, performs its dicided action and obtains the reward. In this example, the Environment would be set as the Gobang board with the size of 8x8. Moreover, to show the location of white chesses and black chesses, its proper to use a 8x8x2 matric to express the states of a game, which is refered from the AlphGo. And there is some attributes of the Environment:

#### 1.Board
A 8x8x2 matric. In the first 8x8 sub-matric, the element is 0 means nothing here while the element is 1 means a black chess here. Constractly in the second 8x8 sub-matric, the element is 0 means nothing here also but the element is 1 means a white chess here insteadly.

#### 2.Perform
A function. Given an action which is a tuple of 3 int, the Environment would always accept this action and update the Board, then return the reward of this action. Importantly, the 3 int, which are the x, y, z, have there own meaning. The x means the distance between the chess and the left-border, while the y means the distance between the chess and the up-border. The x, y are both between 0 and 8(cannot be 8). The z is between 0 and 2(cannot be 2), z is 0 means a black chess while 1 means a white chess.

#### 3.Reward
A int usually. The reward is timely, which should be set carefully. First of all, if the action is not legal(examply place one chess on another chess's location), the reward would be -999 as a punishment so that the agent would not try to do this action. Secondly, if the action win the total game, the reward would be 999 as a conguratulation so that the agent would try to make the state be good for itself. All other situations return a 0 as reward.

#### 4.Terminal
A bool. If the game ends, return the true.

#### 5.Clear
A function. After a game, clear the board, reset all its element as 0 to prepare for the next game.


### Agent
Agent is who uses the Actor-net and Critic-net, performs the action, interacts with Environment and accepts the reward. In one game, the Agent will use its policy to decide the action it would perform under current state, which should be an 8x8x2 matric, so it needs the Actor-net to help it do deployment. Then in every game, the Agent would alawys record the game info which is a trajectory like ${s_{1}, a_{1}, r_{1}, s_{2}}$. There would be a replay memory to store this info, which is named buffer.

#### 1.PlayGround
Just the Environment, input as an instance when the Agent intializing. It serves as the Agent's one member variables.

#### 2.Policy
A function. The Policy function call an Actor-net and at the same time input the current state, the accept the net's return as the guide for its deployment. According to design, the Actor-net would accept an 8x8x2 matric as its input and return an 8x8x2 which show the probability of all possible action. The new matric's sum should be one. And the policy not alawys select the most possible action, it would do a random selection based on the return probability matric. The Policy return a tuple of 3 int as the selected action.

#### 3.Buffer
A list with external storage. Whenever an Agent is created, it should be given a buffer to store its experience. The Buffer has a fixed capital, which would be set as 256 according to disign. To make the buffer reusable, there should be a external file to store the info. Every time the Buffer is created, it first obtain the info in this external file, then do update to it, at last update the external file before the whole program ends.

#### 4.Play
A function, call the agent start interact with PlayGround until the PlayGround return a terminal signal.


### Actor-net
Actor-net is a neral-network based on the module API in pytorch.

- 2 hidden layer(liner + relu);
- a softmax layer;
- $\theta\leftarrow\theta+{\alpha}J(\theta)$

Actor-net accept 8x8x2 matric and return a new 8x8x2 matric.


### Critic-net
Critic-net is a neral-network based on the module API in pytorch.

- a concat option;
- 2 hidden layer(liner + relu);
- $\omega\leftarrow\omega-{\beta}TD-error$

Critic-net accept 8x8x2 matric and a tuple of 3 int the return a float.


### Training Process
- 1.Create an Environment and an Agent instance.
- 2.Let the Agent play the Game for some times.
- 3.Sample form the Buffer, obtain $N\times(s_{t}, a_{t}, r_{t}, s_{t+1})$.
- 4.Update the parameters of the Actor-net and the Critic-net, like this:
    - a) select a 4-int tuple form $N\times(s_{t}, a_{t}, r_{t}, s_{t+1})$, assert it's $(s_{t}, a_{t}, r_{t}, s_{t+1})$
    - b) $a_{t+1}=\pi(·|s_{t+1};\theta)$
    - c) $q_{t}=Q^\pi(s_{t},a_{t};\omega)$ &nbsp; $q_{t+1}=Q^\pi(s_{t+1},a_{t+1};\omega)$
    - d) $\delta_{t}=q_{t}-(r_{t}+{\gamma}q_{t+1})$
    - e) $d_{\theta,t}=\frac{\partial{log{\pi(a_{t}|s_{t};\theta)}}}{\partial{\theta}}$ &nbsp; $d_{\omega,t}=\frac{\partial{Q^\pi(s_{t},a_{t};\omega)}}{\partial{\omega}}$
    - f) $\theta_{new}=\theta_{now}+{\alpha}\cdot{q_{\delta_{t}}}\cdot{d_{\theta,t}}$ &nbsp; $\omega_{new}=\omega_{now}+{\beta}\cdot{q_{\delta_{t}}}\cdot{d_{\omega,t}}$
    - g) repeate options above until the $N\times(s_{t}, a_{t}, r_{t}, s_{t+1})$ all selected.
- 5.Store the 2 module's parameters.



## Method
### Class Environment
```python
class Environment:
    def __init__(self):
        self.board = torch.zeros((2, 8, 8))

    def perform(self, action):
        # place the chess
        z, y, x = action[0], action[1], action[2]
        self.board[z][y][x] += 1

        # return the reward and terminal
        return self.feedback()

    def feedback(self):
        # judge if unlegal or linked
        unlegal = self.unlegal()
        linked = self.linked()

        # caculate reward
        reward = 0
        if unlegal:
            reward = -999
        if linked:
            reward = 999

        # judge if terminal
        terminal = unlegal or linked

        # return result
        return reward, terminal

    def unlegal(self):
        # statistics
        count1 = torch.sum(self.board, 0)
        count2 = torch.sum(self.board, (1,2))
        
        # if a chess conflict
        if 2 in count1:
            return True

        # if order wrong
        delta = count2[0] - count2[1]
        if delta < 0 or delta > 1:
            return True

        # if legal
        return False

    def linked(self):
        board = self.board[0] - self.board[1]

        # judge row
        for i in range(8):
            for j in range(4):
                if board[i][j] == board[i][j+1] == board[i][j+2] == board[i][j+3] == board[i][j+4] != 0:
                    return True

        # judge col
        for j in range(8):
            for i in range(4):
                if board[i][j] == board[i+1][j] == board[i+2][j] == board[i+3][j] == board[i+4][j] != 0:
                    return True

        # judge left>right
        for i in range(4):
            for j in range(4):
                if board[i][j] == board[i+1][j+1] == board[i+2][j+2] == board[i+3][j+3] == board[i+4][j+4] != 0:
                    return True

        # judge left<right
        for i in range(4, 8):
            for j in range(4):
                if board[i][j] == board[i-1][j+1] == board[i-2][j+2] == board[i-3][j+3] == board[i-4][j+4] != 0:
                    return True

        # not linked
        return False

    def clear(self):
        self.board *= 0

    def getState(self):
        return self.board.flatten()
```


### Class Agent
```python
class Agent:
    def __init__(self, env, policy, buffer):
        self.playground = env
        self.policy = policy
        self.buffer = buffer

    def play(self):
        terminal = False
        while(not terminal):
            state1 = self.playground.getState()
            action = self.select(policy(state1))
            reward, terminal = self.playground.perform(action)
            state2 = self.playground.getState()
            self.buffer.update(state1, action, reward, state2)
        self.buffer.store()

    def select(self, guide):
        index = list(range(len(guide)))
        possible = random.choices(index, weights=guide, k=1)[0]

        dim0 = possible // 64
        dim1 = possible % 64 // 8
        dim2 = possible % 64 % 8

        return torch.Tensor([dim0, dim1, dim2]).type(torch.long)
```


### Class Buffer
```python
class Buffer:
    def __init__(self):
        self.capsity = 256
        self.storage = []
        with open('Buffer', 'r') as f:
            for line in f.readlines():
                splitedLine = line.replace('\n','').split(';')
                state1 = splitedLine[0]
                action = splitedLine[1]
                reward = splitedLine[2]
                state2 = splitedLine[3]

                state1 = torch.Tensor([float(ch) for ch in list(state1)])
                action = torch.Tensor([float(ch) for ch in list(action)])
                reward = float(reward)
                state2 = torch.Tensor([float(ch) for ch in list(state2)])

                self.storage.append((state1, action, reward, state2))

    def update(self, state1, action, reward, state2):
        self.storage.append((state1, action, reward, state2))
        if(len(self.storage)>self.capsity):
            self.storage.pop(0)

    def store(self):
        string = ""
        for record in self.storage:
            state1 = record[0]
            action = record[1]
            reward = record[2]
            state2 = record[3]

            state1 = ''.join(str(int(v)) for v in state1)
            action = ''.join(str(int(v)) for v in action)
            reward = str(reward)
            state2 = ''.join(str(int(v)) for v in state2)

            string += ';'.join([state1, action, reward, state2])
            string += '\n'
        with open('Buffer', 'w') as f:
            f.write(string)

    def clear(self):
        string = ""
        with open('Buffer', 'w') as f:
            f.write(string)
```


### Class Actor
```python
class Actor(nn.Module):
   def __init__(self):
       super(Actor, self).__init__()
       self.layers = nn.Sequential(
           nn.Linear(inputDim, 128),
           nn.ReLU(),
           nn.Linear(128, 128),
           nn.ReLU(),
           nn.Linear(128, outputDim))
       self.softmax = nn.Softmax(0) 

   def forward(self, x):
       logits = self.layers(x)
       res = self.softmax(logits)
       return res
```


### Class Critic
```python
class Critic(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(8*8*2+3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):   #Remember concat before input!
        x = self.flatten(x)
        logits = self.layers(x)
        return logits
```


### Func Traing


## Result



## Discussion
