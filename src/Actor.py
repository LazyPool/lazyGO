import torch
from torch import nn

inputDim = 2*8*8
outputDim = 2*8*8

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.flatten = nn.Flatten(0, -1)
        self.layers = nn.Sequential(
            nn.Linear(inputDim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, outputDim))
        self.softmax = nn.Softmax(0) 

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        res = self.softmax(logits)
        return res.reshape(2,8,8)
