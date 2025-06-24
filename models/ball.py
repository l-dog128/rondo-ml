import numpy as np
import math

class Ball:
  def __init__(self,x,y):
    self.pos = np.array([[x,y]],dtype=np.float32)
    # ASSUME the Ball can move at the same speed
    self._maxSpeed = 3
    self.vx = 0
    self.vy = 0
    
    self.target = np.array([None,None],dtype=np.float32)
    self.minDistance = 0.3
    
  def kick(self,targetX, targetY):
    self.target[0] = targetX
    self.target[1] = targetY
    
    difference_vector = self.target - self.pos
    mag = difference_vector[0][0] **2 + difference_vector[0][1] **2
    self.vx = self._maxSpeed * difference_vector[0][0]/mag
    self.vy = self._maxSpeed * difference_vector[0][1]/mag
    
  def step(self):
    self.pos[0][1] += self.vx
    self.pos[0][1] += self.vy
    
    if self.distanceFromTarget() < self.minDistance:
      self.vx = 0
      self.vy = 0

  def distanceFromTarget(self):
    return np.linalg.norm(self.pos-self.target)
  
  def getPos(self):
    return self.pos
    