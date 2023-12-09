import gymnasium
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pygame

import flappy_bird_gymnasium
import random


def play():
    env = gymnasium.make(
        "FlappyBird-v0", audio_on=False, render_mode="human"
    )

    steps = 0

    obs = env.reset()
    while True:
        # Getting action:
        # action = 0
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #     if event.type == pygame.KEYDOWN and (
        #         event.key == pygame.K_SPACE or event.key == pygame.K_UP
        #     ):
        #         action = 1

        action = random.choices([0, 1], weights=[90, 10])[0]
        print(action)

        # Processing:
        obs, _, done, _, info = env.step(action)
        # from IPython import embed; embed()

        steps += 1
        print(
            f"Action: {action}\n"
            f"Score: {info['score']}\n Steps: {steps}\n"
        )

        if done:
            break

    env.close()

if __name__ == "__main__":
    for i in range(100):
        play()