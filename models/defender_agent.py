from collections import defaultdict
import gymnasium as gym
import math
import numpy as np
import torch
from utils import memory_buffer
import models.dqn_network as dqn_network
"""
The agennt implements epsilon-greedy strategy for selecting the best action to choose
"""

class Defender_Agent():
  def __init__(self,
               env:gym.Env,
               batch_size:int,
               gamma:float,
               inital_epsilon:float,
               final_epsilon: float,
               epsilon_decay: float,
               tau:float,
               lr:float,
               ):
    
    self.env = env
    
    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer

    self.batch_size = batch_size
    self.gamma = gamma
    self.eps_start = inital_epsilon
    self.epsilon = inital_epsilon
    self.eps_end = final_epsilon
    self.eps_decay = epsilon_decay
    self.tau = tau
    self.lr = lr
    
    self.n_observations = env.observation_space.shape[0] * 2 # *2 as two cords for each obs
    n_actions = env.action_space.n
    
    # TODO check if we using cuda or not probably not 
    self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")      
    
    # 2 because we end up squishing them all together 
    self.policy_net = dqn_network.DQN(self.n_observations,4).to(self.device)
    self.target_net = dqn_network.DQN(self.n_observations,4).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    
    self.optimizer = torch.optim.AdamW(self.policy_net.parameters(),lr=self.lr,amsgrad=True)
    self.memory = memory_buffer.MemoryBuffer(10_000)
    
    self.steps_done = 0
    
    self.training_error = []
    
  def get_action(self,obs):
    self.steps_done += 1
    if np.random.random() > self.epsilon :
      # TODO this might be a bug but we will see 
      # obs = np.concatenate(list(obs.values()))
      obs = np.array(obs).flatten()
      obs = torch.tensor(obs,dtype=torch.float32,device=self.device).unsqueeze(0)
      # print(obs)
      # print(obs.shape)
      with torch.no_grad():
        # print(self.policy_net(obs))
        q_values = self.policy_net(obs)
        # print(q_values)
        # print(q_values.max(1))
        # print(q_values.max(1).indices.max())
        # print("network")
        return q_values.max(1).indices.max()
    else:
      # print("random")
      return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        
  def update(self,batch_size):
    
    if len(self.memory) < batch_size: return
    
    next_batch = self.memory.sample(batch_size)
    state_batch,next_state_batch,reward_batch,terminated_batch,action_batch = zip(*next_batch)
    
    state_batch = np.array(state_batch).flatten().reshape(10,batch_size).transpose()
    next_state_batch = np.array(next_state_batch).flatten().reshape(10,batch_size).transpose()
    
    state_batch = torch.FloatTensor(state_batch).to(self.device)
    next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
    reward_batch = torch.FloatTensor(reward_batch).to(self.device)
    terminated_batch = torch.FloatTensor(terminated_batch).to(self.device)
    action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
        
    
    q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()
    
    # Compute target Q-values using the target network
    with torch.no_grad():
        max_next_q_values = self.target_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - terminated_batch)

    loss = torch.nn.MSELoss()(q_values, target_q_values)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
      
    # only add training error 10th of the time 
    if np.random.random() < 0.1:
      self.training_error.append(loss.item())
    
  def decay_epsilon(self):
    # TODO see which function is better 
    # self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
    
  def update_targetNet(self):
    self.target_net.load_state_dict(self.policy_net.state_dict())
  