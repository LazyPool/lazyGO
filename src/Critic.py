import torch
from torch import nn

inputDim = 2*8*8+3
outputDim = 1

class Critic(nn.Moudule):
    def __init():
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputDim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, outputDim))
    
    def forward(self, x):
        logits = self.layers(x)
        res = logits
        return res
