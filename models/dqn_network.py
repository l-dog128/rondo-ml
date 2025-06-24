import torch
import torch.nn as nn 
import torch.nn.functional as F


class DQN(nn.Module):
  def __init__(self,inputDim,outputDim):
    super(DQN,self).__init__()
    self.fc1 = nn.Linear(inputDim,128)
    self.fc2 = nn.Linear(128,128)
    self.out = nn.Linear(128,outputDim)
    
  def forward(self,x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.out(x)