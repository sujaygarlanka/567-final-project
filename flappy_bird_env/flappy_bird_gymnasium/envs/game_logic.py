# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
# Copyright (c) 2023 Martin Kubovcik
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Implements the logic of the Flappy Bird game.

Some of the code in this module is an adaption of the code in the `FlapPyBird`
GitHub repository by `sourahbhv` (https://github.com/sourabhv/FlapPyBird),
released under the MIT license.
"""


from enum import IntEnum
from itertools import cycle
from typing import Dict, Tuple, Union

import numpy as np
import pygame

from flappy_bird_gymnasium.envs.constants import (
    BACKGROUND_WIDTH,
    BASE_WIDTH,
    LIDAR_MAX_DISTANCE,
    PIPE_HEIGHT,
    PIPE_VEL_X,
    PIPE_WIDTH,
    PLAYER_ACC_Y,
    PLAYER_FLAP_ACC,
    PLAYER_HEIGHT,
    PLAYER_MAX_VEL_Y,
    PLAYER_VEL_ROT,
    PLAYER_WIDTH,
)
from flappy_bird_gymnasium.envs.lidar import LIDAR


class FlappyBirdLogic:
    """Handles the logic of the Flappy Bird game.

    The implementation of this class is decoupled from the implementation of the
    game's graphics. This class implements the logical portion of the game.

    Args:
        screen_size (Tuple[int, int]): Tuple with the screen's width and height.
        pipe_gap_size (int): Space between a lower and an upper pipe.

    Attributes:
        player_x (int): The player's x position.
        player_y (int): The player's y position.
        ground (dict): The base/ground's x and y positions.
        score (int): Current score of the player.
        upper_pipes (List[Dict[str, int]): List with the upper pipes. Each pipe
            is represented by a dictionary containing two keys: "x" (the pipe's
            x position) and "y" (the pipe's y position).
        lower_pipes (List[Dict[str, int]): List with the lower pipes. Each pipe
            is represented by a dictionary containing two keys: "x" (the pipe's
            x position) and "y" (the pipe's y position).
        player_vel_y (int): The player's vertical velocity.
        player_rot (int): The player's rotation angle.
        sound_cache (Optional[str]): Stores the name of the next sound to be
            played. If `None`, then no sound should be played.
        player_idx (int): Current index of the bird's animation cycle.
    """

    def __init__(
        self,
        np_random,
        screen_size: Tuple[int, int],
        pipe_gap_size: int = 100,
        use_lidar=True,
    ) -> None:
        self._screen_width = screen_size[0]
        self._screen_height = screen_size[1]

        self.player_x = int(self._screen_width * 0.2)
        self.player_y = int((self._screen_height - PLAYER_HEIGHT) / 2)

        self.ground = {"x": 0, "y": self._screen_height * 0.79}
        self._base_shift = BASE_WIDTH - BACKGROUND_WIDTH

        self.score = 0
        self._pipe_gap_size = pipe_gap_size

        self._np_random = np_random

        # Generate 3 new pipes to add to upper_pipes and lower_pipes lists
        new_pipe1 = self._get_random_pipe()
        new_pipe2 = self._get_random_pipe()
        new_pipe3 = self._get_random_pipe()

        # List of upper pipes:
        self.upper_pipes = [
            {"x": self._screen_width, "y": new_pipe1[0]["y"]},
            {
                "x": self._screen_width + (self._screen_width / 2),
                "y": new_pipe2[0]["y"],
            },
            {
                "x": self._screen_width + self._screen_width,
                "y": new_pipe3[0]["y"],
            },
        ]

        # List of lower pipes:
        self.lower_pipes = [
            {"x": self._screen_width, "y": new_pipe1[1]["y"]},
            {
                "x": self._screen_width + (self._screen_width / 2),
                "y": new_pipe2[1]["y"],
            },
            {
                "x": self._screen_width + self._screen_width,
                "y": new_pipe3[1]["y"],
            },
        ]

        # Player's info:
        self.player_vel_y = -9  # player"s velocity along Y
        self.player_rot = 45  # player"s rotation

        self.sound_cache = None
        self._player_flapped = False
        self.player_idx = 0
        self._player_idx_gen = cycle([0, 1, 2, 1])
        self._loop_iter = 0

        if use_lidar:
            self.lidar = LIDAR(LIDAR_MAX_DISTANCE)
            self.get_observation = self.get_observation_lidar
        else:
            self.get_observation = self._get_observation_features

    class Actions(IntEnum):
        """Possible actions for the player to take."""

        IDLE, FLAP = 0, 1

    def _get_random_pipe(self) -> Dict[str, int]:
        """Returns a randomly generated pipe."""
        # y of gap between upper and lower pipe
        gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
        index = self._np_random.integers(0, len(gapYs))
        gap_y = gapYs[index]
        gap_y += int(self.ground["y"] * 0.2)

        pipe_x = self._screen_width + PIPE_WIDTH + (self._screen_width * 0.2)
        return [
            {"x": pipe_x, "y": gap_y - PIPE_HEIGHT},  # upper pipe
            {"x": pipe_x, "y": gap_y + self._pipe_gap_size},  # lower pipe
        ]

    def check_crash(self) -> bool:
        """Returns True if player collides with the ground (base) or a pipe."""
        # if player crashes into ground
        if self.player_y + PLAYER_HEIGHT >= self.ground["y"] - 1:
            return True
        elif self.player_y <= 0:
            return True
        else:
            player_rect = pygame.Rect(
                self.player_x, self.player_y, PLAYER_WIDTH, PLAYER_HEIGHT
            )

            for up_pipe, low_pipe in zip(self.upper_pipes, self.lower_pipes):
                # upper and lower pipe rects
                up_pipe_rect = pygame.Rect(
                    up_pipe["x"], up_pipe["y"], PIPE_WIDTH, PIPE_HEIGHT
                )
                low_pipe_rect = pygame.Rect(
                    low_pipe["x"], low_pipe["y"], PIPE_WIDTH, PIPE_HEIGHT
                )

                # check collision
                up_collide = player_rect.colliderect(up_pipe_rect)
                low_collide = player_rect.colliderect(low_pipe_rect)

                if up_collide or low_collide:
                    return True

        return False

    def _get_observation_features(self, normalize: bool = True) -> np.ndarray:
        pipes = []
        for up_pipe, low_pipe in zip(self.upper_pipes, self.lower_pipes):
            # the pipe is behind the screen?
            if low_pipe["x"] > self._screen_width:
                pipes.append((self._screen_width, 0, self._screen_height))
            else:
                pipes.append(
                    (low_pipe["x"], (up_pipe["y"] + PIPE_HEIGHT), low_pipe["y"])
                )

        pipes = sorted(pipes, key=lambda x: x[0])
        pos_y = self.player_y
        vel_y = self.player_vel_y
        rot = self.player_rot

        if normalize:
            pipes = [
                (
                    h / self._screen_width,
                    v1 / self._screen_height,
                    v2 / self._screen_height,
                )
                for h, v1, v2 in pipes
            ]
            pos_y = pos_y / self._screen_height
            vel_y /= PLAYER_MAX_VEL_Y
            rot /= 90

        # return np.array(
        #     [
        #         pipes[0][0],  # the last pipe's horizontal position
        #         pipes[0][1],  # the last top pipe's vertical position
        #         pipes[0][2],  # the last bottom pipe's vertical position
        #         pipes[1][0],  # the next pipe's horizontal position
        #         pipes[1][1],  # the next top pipe's vertical position
        #         pipes[1][2],  # the next bottom pipe's vertical position
        #         pipes[2][0],  # the next next pipe's horizontal position
        #         pipes[2][1],  # the next next top pipe's vertical position
        #         pipes[2][2],  # the next next bottom pipe's vertical position
        #         pos_y,  # player's vertical position
        #         vel_y,  # player's vertical velocity
        #         rot,  # player's rotation
        #     ]
        # )

        return np.array(
            [
                pipes[0][0],  # the last pipe's horizontal position
                pipes[0][1],  # the last top pipe's vertical position
                pipes[0][2],  # the last bottom pipe's vertical position
                pos_y,  # player's vertical position
                vel_y,  # player's vertical velocity
            ]
        )
        

    def get_observation_lidar(self, normalize=True):
        # obstacles
        distances = self.lidar.scan(
            self.player_x,
            self.player_y,
            self.player_rot,
            self.upper_pipes,
            self.lower_pipes,
            self.ground,
            normalize,
        )
        return distances

    def update_state(self, action: Union[Actions, int], normalize=True) -> bool:
        """Given an action taken by the player, updates the game's state.

        Args:
            action (Union[FlappyBirdLogic.Actions, int]): The action taken by
                the player.

        Returns:
            `True` if the player is alive and `False` otherwise.
        """
        reward = 0.1  # reward for staying alive
        terminal = False

        self.sound_cache = None
        if action == FlappyBirdLogic.Actions.FLAP:
            if self.player_y > -2 * PLAYER_HEIGHT:
                self.player_vel_y = PLAYER_FLAP_ACC
                self._player_flapped = True
                self.sound_cache = "wing"

        # check for score
        player_mid_pos = self.player_x + PLAYER_WIDTH / 2
        for pipe in self.upper_pipes:
            pipe_mid_pos = pipe["x"] + PIPE_WIDTH / 2
            if pipe_mid_pos <= player_mid_pos < pipe_mid_pos + 4:
                self.score += 1
                reward = 1  # reward for passed pipe
                self.sound_cache = "point"

        # player_index base_x change
        if (self._loop_iter + 1) % 3 == 0:
            self.player_idx = next(self._player_idx_gen)

        self._loop_iter = (self._loop_iter + 1) % 30
        self.ground["x"] = -((-self.ground["x"] + 100) % self._base_shift)

        # rotate the player
        if self.player_rot > -90:
            self.player_rot -= PLAYER_VEL_ROT

        # player's movement
        if self.player_vel_y < PLAYER_MAX_VEL_Y and not self._player_flapped:
            self.player_vel_y += PLAYER_ACC_Y

        if self._player_flapped:
            self._player_flapped = False

            # more rotation to cover the threshold
            # (calculated in visible rotation)
            self.player_rot = 45

        self.player_y += min(
            self.player_vel_y, self.ground["y"] - self.player_y - PLAYER_HEIGHT
        )

        # agent touch the top of the screen as punishment
        if self.player_y < 0:
            reward = -0.5

        # move pipes to left
        for up_pipe, low_pipe in zip(self.upper_pipes, self.lower_pipes):
            up_pipe["x"] += PIPE_VEL_X
            low_pipe["x"] += PIPE_VEL_X

            # it is out of the screen
            if up_pipe["x"] < -PIPE_WIDTH:
                new_up_pipe, new_low_pipe = self._get_random_pipe()
                up_pipe["x"] = new_up_pipe["x"]
                up_pipe["y"] = new_up_pipe["y"]
                low_pipe["x"] = new_low_pipe["x"]
                low_pipe["y"] = new_low_pipe["y"]

        # check for crash
        if self.check_crash():
            self.sound_cache = "hit"
            reward = -1  # reward for dying
            terminal = True
            self.player_vel_y = 0

        return self.get_observation(normalize=normalize), reward, terminal
