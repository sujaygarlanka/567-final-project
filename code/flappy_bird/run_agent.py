import time
import flappy_bird_gym
import sys

sys.path.append("../")
from lib.matrix_approach.layers import *

env = flappy_bird_gym.make("FlappyBird-v0", pipe_gap = 150)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

class NN(Network):
    def __init__(self):
        self.layers = [
            FC(state_size, 64),
            ReLU(),
            FC(64, 128),
            ReLU(),
            FC(128, action_size),
        ]

    def __call__(self, inputs):
        input = np.array(inputs)
        return self.forward(input)

model = NN()
model.load("models/flappy_bird.npy")

state = env.reset()
num = 0
while True:
    # Predict best action:
    q_values = model([state])
    action = np.argmax(q_values[0])

    # Take action:
    state, reward, done, info = env.step(action)

    # Rendering the game:
    # (remove this two lines during training)
    env.render()
    time.sleep(1 / 30)  # FPS
    num += 1

    # Checking if the player is still alive
    if done:
        print(num)
        break

env.close()
