import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
'''
this is the classifier model with 
    3 linear layers 
'''
class classifire(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=10),
        )
    
    def forward(self, x:torch.Tensor):
        x = self.block(x)
        return x
