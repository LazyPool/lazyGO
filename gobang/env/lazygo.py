import gym
from gym import spaces
import pygame
import numpy as np



class lazyGO(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.size = 8
        self.windowsize = 512

        self.observation_space = spaces.MultiDiscrete(np.ones((self.size, self.size))*3)
        self.action_space = spaces.Discrete(self.size**2)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._board = np.array([[0 for c in range(self.size)] for r in range(self.size)])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info


    def step(self, action):
        row, col = action//self.size, action%self.size

        try:
            assert self._board[row][col] == 0, "cannot move"
        except AssertionError:
            return self._get_obs(), -999, False, False, {}

        self._board[row][col] = 1

        while True:
            row = self.np_random.integers(0, self.size, size=1, dtype=int).item()
            col = self.np_random.integers(0, self.size, size=1, dtype=int).item()
            if self._board[row][col] == 0: break

        self._board[row][col] = 2

        observation = self._get_obs()
        reward, terminated = self._get_check()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


    def _get_obs(self):
        return self._board


    def _get_check(self):
        if self._linked(1):
            reward, terminated = 10, True
        elif self._linked(2):
            reward, terminated = -10, True
        elif self._isfull():
            reward, terminated = 0, True
        else:
            reward, terminated = 0, False

        return reward, terminated

    
    def _get_info(self):
        return {}


    def _linked(self, who):
        board = self._board

        for i in range(8):
            for j in range(4):
                if board[i][j] == board[i][j+1] == board[i][j+2] == board[i][j+3] == board[i][j+4] == who:
                    return True

        for j in range(8):
            for i in range(4):
                if board[i][j] == board[i+1][j] == board[i+2][j] == board[i+3][j] == board[i+4][j] == who:
                    return True

        for i in range(4):
            for j in range(4):
                if board[i][j] == board[i+1][j+1] == board[i+2][j+2] == board[i+3][j+3] == board[i+4][j+4] == who:
                    return True

        for i in range(4, 8):
            for j in range(4):
                if board[i][j] == board[i-1][j+1] == board[i-2][j+2] == board[i-3][j+3] == board[i-4][j+4] == who:
                    return True


    def _isfull(self):
        if 0 not in self._board:
            return True
        else:
            return False


    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.windowsize, self.windowsize))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.windowsize, self.windowsize))
        canvas.fill((255,255,255))
        pix_square_size = self.windowsize / self.size

        for i in range(self.size):
            for j in range(self.size):
                if self._board[i][j] == 1:
                    pygame.draw.circle(
                        canvas,
                        (0, 0, 0),
                        ((i+0.5) * pix_square_size, (j+0.5) * pix_square_size),
                        pix_square_size / 3,
                    )
                elif self._board[i][j] == 2:
                    pygame.draw.circle(
                        canvas,
                        (255, 0, 0),
                        ((i+0.5) * pix_square_size, (j+0.5) * pix_square_size),
                        pix_square_size / 3,
                    )
                else:
                    pass
        
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.windowsize, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.windowsize),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas), axes=(1,0,2))
            )

