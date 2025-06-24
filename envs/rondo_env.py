from typing import Optional
import gymnasium as gym
import numpy as np
import pygame
from models import ball
from models import attacker

class RondoEnv(gym.Env):
  metadata = {"render_modes": ["human","rgb_array"],"render_fps": 10}
  def __init__(self,size=7,attackers=3,render_mode=None,max_steps = 200):
    super(RondoEnv,self).__init__()
    
    self.size = size
    self.window_size = 720
    
    self.step_num = 0
    self.max_steps = max_steps
    self.min_dist_ball = 0.8
    
    self.attackers = np.array([None]*attackers,dtype=attacker.Attacker)
    self._attacker_locations = np.array([-1,-1]*attackers,dtype=np.float32).reshape(attackers,2)
    
    self.ball = None
    
    self._agent_location = np.array([size//2-1,size//2-1])
    
    # change the observation space 
    total_params = 2+attackers 
    self.observation_space = gym.spaces.Box(0,size-1,shape=(5,2),dtype=np.float32)
    
    # self.observation_space = gym.spaces.Dict({
    #  "agent" : gym.spaces.Box(0,size-1,shape=(2,),dtype=np.float32),
    #  "attackers_pos" : gym.spaces.Box(0,size-1,shape=(3,2),dtype=np.float32),
    #  "ball_pos": gym.spaces.Box(0,size-1,shape=(2,),dtype=np.float32)
    # })
    
    # We have 4 actions, corresponding to "right", "up", "left", "down" 
    # TODO maybe change to vector space so we can move any direction
    self.action_space = gym.spaces.Discrete(4)
    # Dictionary maps the abstract actions to the directions on the grid
    self._action_to_direction = {
      0: np.array([1, 0]),  # right
      1: np.array([0, 1]),  # up
      2: np.array([-1, 0]),  # left
      3: np.array([0, -1]),  # down
    }
    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode
    
    # pygame stuff
    self.window = None
    self.clock = None
    
  def _get_obs(self):
    # TODO think about what the AGENT Sees
    self._agent_location = self._agent_location.reshape((1,2))
    
    return np.concatenate((self._agent_location,self._attacker_locations,self.ball.getPos()))
    return {"agent":self._agent_location,"attackers_pos":self._attacker_locations,"ball_pos":self.ball.getPos()}
  
  def _get_info(self):
    # TODO think about what to return here 
    return{}
    
  def _get_reward(self):
    if self.distanceFromBall() < self.min_dist_ball: return 1
    return 0
  
  def reset(self,seed:Optional[int]=None,options:Optional[dict]=None):
    # IMPORTANT!
    super().reset(seed=seed)
    
    num_attackers = len(self._attacker_locations) 
    radius = 4
    
    # create the ball 
    theta = 2*np.pi/num_attackers
    self.ball = ball.Ball(radius *  np.cos(theta) + self.size//2,radius *  np.sin(theta) + self.size//2 )
    
    # set attacker posistions and create them
    for i in range(0,num_attackers):
      theta = 2*np.pi*(i+1)/num_attackers
      x = radius *  np.cos(theta) + self.size//2
      y = radius *  np.sin(theta) + self.size//2 
      self.attackers[i] = attacker.Attacker(x,y,self.ball)
      self._attacker_locations[i][0] = int(radius *  np.cos(theta) + self.size//2)
      self._attacker_locations[i][1] = int(radius *  np.sin(theta) + self.size//2)
      
    for atcker in self.attackers:
      for otherAttacker in self.attackers:
        if(atcker != otherAttacker):
          atcker.addOtherPlayer(otherAttacker)
      
    self._agent_location = np.array([self.size//2-1,self.size//2-1],dtype=np.float32)
    self.step_num = 0
    observation = self._get_obs()
    info = self._get_info()
     
    return observation,info
  
  def step(self,action):
    direction = self._action_to_direction[action]
    self._agent_location = np.clip(self._agent_location + direction,0,self.size-1,dtype=np.float32)
    
    for attacker in self.attackers:
      attacker.step()  
    self.ball.step()
    
    # check if player has intercepted the ball
    terminated = self.distanceFromBall() < self.min_dist_ball
    truncated = False
    if self.step_num > self.max_steps:
      truncated = True
    
    reward = self._get_reward()
    next_obs = self._get_obs()
    info = self._get_info()
        
    if self.render_mode == "human":
        self._render_frame()
      
    self.step_num += 1
    return next_obs,reward,terminated,truncated,info
    
  def render(self):
    if self.render_mode == "rgb_array":
      return self._render_frame()
  
  def _render_frame(self):
    if self.render_mode == "human" and self.window is None:
      pygame.init()
      pygame.display.init()
      self.window = pygame.display.set_mode(
        (self.window_size,self.window_size)
      )
    if self.clock is None and self.render_mode == "human":
      self.clock = pygame.time.Clock()
    
    canvas = pygame.Surface((self.window_size,self.window_size))
    canvas.fill((40, 128, 20))
    pix_square_size = int(
      self.window_size / self.size
    )  # The size of a single grid square in pixels
    
    # Now we draw the agent
    pygame.draw.circle(
        canvas,
        (100, 2, 2),
        (self._agent_location + 0.5) * pix_square_size,
        pix_square_size / 3,
    )
    # Now the attackers 
    for attacker in self._attacker_locations:
      pygame.draw.circle(
        canvas,
        (10,10,225),
        # (attacker[0],attacker[1]),
        ((attacker[0] + 0.5) * pix_square_size,(attacker[1] + 0.5) * pix_square_size),
        pix_square_size / 3
      )
    # now the ball
    pygame.draw.circle(
      canvas,
      (255,255,255),
      ((self.ball.pos[0] +0.5) * pix_square_size,(self.ball.pos[1] + 0.5) * pix_square_size),
      pix_square_size / 4
    )
    
    if self.render_mode == "human":
      self.window.blit(canvas,canvas.get_rect())
      pygame.event.pump()
      pygame.display.update()
      self.clock.tick(self.metadata["render_fps"])
    else:
      return np.transpose(
        np.array(pygame.surfarray.pixels3d(canvas)),axes=(1,0,2)
      )
  def close(self):
    if self.window is not None:
        pygame.display.quit()
        pygame.quit()    
        
  def distanceFromBall(self):
    return np.linalg.norm(self.ball.pos - self._agent_location)
