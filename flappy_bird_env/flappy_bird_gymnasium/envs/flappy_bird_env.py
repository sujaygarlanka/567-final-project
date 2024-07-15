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

""" Implementation of a Flappy Bird OpenAI gymnasium environment that yields simple
numerical information about the game's state as observations.
"""

import time
from typing import Dict, Optional, Tuple, Union

import gymnasium
import numpy as np
import pygame

from flappy_bird_gymnasium.envs.constants import LIDAR_MAX_DISTANCE
from flappy_bird_gymnasium.envs.game_logic import FlappyBirdLogic
from flappy_bird_gymnasium.envs.renderer import FlappyBirdRenderer


class FlappyBirdEnv(gymnasium.Env):
    """Flappy Bird Gymnasium environment that yields simple observations.

    The observations yielded by this environment are simple numerical
    information about the game's state. Specifically, the observations are:

        * Horizontal distance to the next pipe;
        * Difference between the player's y position and the next hole's y
          position.

    The reward received by the agent in each step is equal to the score obtained
    by the agent in that step. A score point is obtained every time the bird
    passes a pipe.

    Args:
        screen_size (Tuple[int, int]): The screen's width and height.
        normalize_obs (bool): If `True`, the observations will be normalized
            before being returned.
        pipe_gap (int): Space between a lower and an upper pipe.
        bird_color (str): Color of the flappy bird. The currently available
            colors are "yellow", "blue" and "red".
        pipe_color (str): Color of the pipes. The currently available colors are
            "green" and "red".
        background (Optional[str]): Type of background image. The currently
            available types are "day" and "night". If `None`, no background will
            be drawn.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        screen_size: Tuple[int, int] = (288, 512),
        audio_on: bool = False,
        normalize_obs: bool = True,
        use_lidar: bool = False,
        pipe_gap: int = 100,
        bird_color: str = "yellow",
        pipe_color: str = "green",
        render_mode: Optional[str] = None,
        background: Optional[str] = "day",
    ) -> None:
        self.action_space = gymnasium.spaces.Discrete(2)
        if use_lidar:
            self.observation_space = gymnasium.spaces.Box(
                -np.inf, np.inf, shape=(180,), dtype=np.float64
            )
        else:
            self.observation_space = gymnasium.spaces.Box(
                -np.inf, np.inf, shape=(5,), dtype=np.float64
            )
        self._fps_clock = pygame.time.Clock()
        self._screen_size = screen_size
        self._normalize_obs = normalize_obs
        self._pipe_gap = pipe_gap
        self._audio_on = audio_on
        self._use_lidar = use_lidar

        self._game = None
        self._renderer = None

        self._bird_color = bird_color
        self._pipe_color = pipe_color
        self._bg_type = background

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if render_mode is not None:
            self._renderer = FlappyBirdRenderer(
                screen_size=self._screen_size,
                audio_on=audio_on,
                bird_color=bird_color,
                pipe_color=pipe_color,
                background=background,
            )

    def step(
        self,
        action: Union[FlappyBirdLogic.Actions, int],
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Given an action, updates the game state.

        Args:
            action (Union[FlappyBirdLogic.Actions, int]): The action taken by
                the agent. Zero (0) means "do nothing" and one (1) means "flap".

        Returns:
            A tuple containing, respectively:

                * an observation (horizontal distance to the next pipe
                  difference between the player's y position and the next hole's
                  y position)
                * a reward (alive = +0.1, pipe = +1.0, dead = -1.0)
                * a status report (`True` if the game is over and `False`
                  otherwise)
                * an info dictionary
        """
        obs, reward, terminal = self._game.update_state(action, self._normalize_obs)
        info = {"score": self._game.score}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminal, False, info

    def reset(self, seed=None, options=None):
        """Resets the environment (starts a new game)."""
        super().reset(seed=seed)

        self._game = FlappyBirdLogic(
            np_random=self.np_random,
            screen_size=self._screen_size,
            pipe_gap_size=self._pipe_gap,
            use_lidar=self._use_lidar,
        )
        if self._renderer is not None:
            self._renderer.game = self._game

        if self.render_mode == "human":
            self.render()

        obs = self._game.get_observation(self._normalize_obs)
        info = {"score": self._game.score}
        return obs, info

    def set_color(self, color):
        if self._renderer is not None:
            self._renderer.set_color(color)

    def render(self) -> None:
        """Renders the next frame."""
        if self.render_mode == "rgb_array":
            self._renderer.draw_surface(show_score=False, show_rays=False)
            # Flip the image to retrieve a correct aspect
            return np.transpose(
                pygame.surfarray.array3d(self._renderer.surface), axes=(1, 0, 2)
            )
        else:
            self._renderer.draw_surface(show_score=True, show_rays=self._use_lidar)
            if self._renderer.display is None:
                self._renderer.make_display()

            self._renderer.update_display()
            self._fps_clock.tick(self.metadata["render_fps"])

    def close(self):
        """Closes the environment."""
        if self._renderer is not None:
            pygame.display.quit()
            pygame.quit()
            self._renderer = None
        super().close()
