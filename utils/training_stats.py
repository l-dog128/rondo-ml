# TAKEN and ADAPTED from "https://gymnasium.farama.org/introduction/train_agent/" 

import numpy as np 
from matplotlib import pyplot as plt

def showStatistics(env,agent):
  def get_moving_avgs(arr, window, convolution_mode):
      return np.convolve(
          np.array(arr).flatten(),
          np.ones(window),
          mode=convolution_mode
      ) / window

  # Smooth over a 25 episode window
  rolling_length = 25
  fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

  axs[0].set_title("Episode rewards")
  reward_moving_average = get_moving_avgs(
      env.return_queue,
      rolling_length,
      "valid"
  )
  axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

  axs[1].set_title("Episode lengths")
  length_moving_average = get_moving_avgs(
      env.length_queue,
      rolling_length,
      "valid"
  )
  axs[1].plot(range(len(length_moving_average)), length_moving_average)

  axs[2].set_title("Training Error")
  training_error_moving_average = get_moving_avgs(
      agent.training_error,
      rolling_length,
      "same"
  )
  axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
  plt.tight_layout()
  plt.show()