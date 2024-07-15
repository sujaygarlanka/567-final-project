import time
import sys
import flappy_bird_gymnasium
import gymnasium as gym 

sys.path.append("../")
from lib.matrix_approach.layers import *
from IPython import display, embed
import matplotlib.pyplot as plt

env = gym.make("FlappyBird-v0", render_mode="human", pipe_gap=150)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# class NN(Network):
#     def __init__(self):
#         self.layers = [
#             FC(state_size, 64),
#             ReLU(),
#             FC(64, 128),
#             ReLU(),
#             FC(128, action_size),
#         ]

#     def __call__(self, inputs):
#         input = np.array(inputs)
#         return self.forward(input)


# model = NN()
# model.load("flappy_bird_model.npy")

num = 0
state, _ = env.reset()
while True:
    # Predict best action:
    # q_values = model([state])
    # action = np.argmax(q_values[0])
    action = env.action_space.sample()

    # Take action:
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    num += 1
    # Checking if the player is still alive
    if done:
        break

env.close()