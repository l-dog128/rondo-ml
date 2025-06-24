from collections import deque
import random

class MemoryBuffer(object):
  def __init__(self,capacity):
    self.memory = deque([],capacity)
    
  def push(self,obs):
    self.memory.append(obs)
    
  def sample(self,batch_size):
    return random.sample(self.memory,batch_size)
  
  def __len__ (self):
    return len(self.memory)