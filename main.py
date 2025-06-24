import numpy as np 
import gymnasium as gym
import envs
from tqdm import tqdm
from utils import training_stats
from matplotlib import pyplot as plt
from models import defender_agent
import logging

N_EPISODES =  10_000 + 1
TRAINING_PERIOD = 20_000
# constants hyperparameters
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
# TARGET_UPDATE_PERDIOD is the rate at which the target network is updated

BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = EPS_START / (N_EPISODES / 2)
TAU = 0.005
LR = 3e-4
TARGET_UPDATE_PERDIOD = 512


env = gym.make("Rondo-V0",render_mode=None,size=11)
# env = gym.wrappers.RecordVideo(env, video_folder="rondo_training", name_prefix="training",
#                   episode_trigger=lambda x: x % TRAINING_PERIOD == 0)
env = gym.wrappers.RecordEpisodeStatistics(env)

agent = defender_agent.Defender_Agent(env,BATCH_SIZE,GAMMA,EPS_START,EPS_END,EPS_DECAY,TAU,LR)

step_count = 0

for episode in tqdm(range(N_EPISODES)):
    
  observation, info = env.reset()
  episode_over = False

  while not episode_over:
    action = agent.get_action(observation)  # agent policy that uses the observation and info
    # print(action)
    next_observation, reward, terminated, truncated, info = env.step(action.item())
    

    agent.memory.push((observation,next_observation, reward, terminated * truncated,action))
    
    episode_over = terminated or truncated
    observation = next_observation
    
    agent.update(BATCH_SIZE)
    
    if(step_count % TARGET_UPDATE_PERDIOD == 0):
      agent.update_targetNet()
    
          
          
      
  agent.decay_epsilon()

env.close()
training_stats.showStatistics(env,agent)

