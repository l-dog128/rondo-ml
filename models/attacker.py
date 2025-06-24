import math
import random
import numpy as np 

class Attacker:
  def __init__(self,x,y,ball: type):
    self.pos = [x,y]
    self.otherPlayers = []
    
    self.ball = ball
  
  def addOtherPlayer(self,player):
    self.otherPlayers.append(player)
  
  def kick(self,targetPlayer):
    self.ball.kick(targetPlayer.pos[0],targetPlayer.pos[1])
      
  def step(self):
    # TODO kick to avoid the defender right now just random teammate 
     if self.distanceFromBall() < self.ball.minDistance:
      targetPlayer = random.sample(self.otherPlayers,1)[0]
      self.kick(targetPlayer)
    
  def distanceFromBall(self):
    return np.linalg.norm(self.ball.pos - self.pos)
    
    
    