import time
import flappy_bird_gym
import sys

sys.path.append("../")
from lib.matrix_approach.layers import *
from IPython import display, embed
import matplotlib.pyplot as plt
from PIL import Image

env = flappy_bird_gym.make("FlappyBird-rgb-v0", pipe_gap=150, screen_size=(288, 512))
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


def rgb2gray(rgb):
    return np.dot(rgb[:, :, :3], [0.2989, 0.5870, 0.1140]) / 255.0


def crop(grayscale):
    # Crop image from 288x512 to 84x84
    return grayscale[102:186, 214:298]


state = env.reset()
print(state)
im = Image.fromarray(state)
im = im.crop((102, 186, 214, 298))
im = im.resize((84, 84))
im.show()
# state = crop(state)
# print(state.shape)
# plt.imshow(state, cmap="gray")
# plt.show()
# from IPython import embed; embed()
num = 0
while True:
    # Predict best action:
    # q_values = model([state])
    # action = np.argmax(q_values[0])
    action = env.action_space.sample()

    # Take action:
    state, reward, done, info = env.step(action)

    # Rendering the game:
    # (remove this two lines during training)
    img.set_data(state)  # just update the data
    plt.show()
    time.sleep(1 / 30)  # FPS
    num += 1

    # Checking if the player is still alive
    if done:
        break

env.close()
