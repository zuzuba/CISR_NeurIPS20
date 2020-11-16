import sys
from contextlib import closing

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import utils
from gym.envs.toy_text import discrete
from gym.envs.toy_text.frozen_lake import generate_random_map
from six import StringIO

from src.envs.frozen_lake.flake_constants import LEFT, DOWN, RIGHT, UP, \
    REWARD_MAPPING, TERMINAL_STATES, NOT_SLIPPING_PROB
from src.envs.frozen_lake.frozen_maps import MAPS
from src.envs.frozen_lake.utils import create_reward_shaping_map

__all__ = ['FrozenLakeEnvCustom', 'FrozenLakeEnvCustomMap', 'fallen_in_lake']


class DiscreteEnvCustom(discrete.DiscreteEnv):

    """
    Modfication of the original discrete env to return the info of transitions
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary over states containinng dictionaries over actions. Each one contains
      P[s][a] == {'p': [probability], 'transition': [(next_s, reward, done, info)]} The value of 'p' is a list with the
      probability over the transitions. Transition is a list of tuples. each tuple contains the full info that is the
      output of an openai gym.step env.
    (**) list or array of length nS
    """

    def __init__(self, nS, nA, P, isd, timeout=np.inf):
        self.nsteps = 0
        self.timeout = timeout
        super(DiscreteEnvCustom, self).__init__(nS, nA, P, isd)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = discrete.categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d, info = transitions[i]
        self.s = s
        self.lastaction = a
        info.update({"prob": p})
        self.nsteps += 1
        d = np.logical_or(d, self.nsteps > self.timeout)
        return s, r, d, info

    def reset(self):
        self.nsteps = 0
        return super(DiscreteEnvCustom, self).reset()


class FrozenLakeEnvCustom(DiscreteEnvCustom):
    """
    Modification of Frozen Lake that can set probability of not slipping, records cell type and can modify rewards.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4", is_slippery=True,
                 not_slipping_prob=None, base_r_mapping=None, timeout=np.inf,
                 shaping_coef=0, shaping_potential=None, n_velocities=1):
        # Get map info
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = None

        # Set slipping prob for the dynamics
        if not_slipping_prob is not None:
            self.is_slippery = True
            self.not_slipping_prob = not_slipping_prob
        else:
            self.is_slippery = is_slippery
            self.not_slipping_prob = NOT_SLIPPING_PROB

        # Set basic reward
        self.base_r = base_r_mapping if base_r_mapping is not None else REWARD_MAPPING

        # Dimensionality
        nA = 4 * n_velocities
        nS = nrow * ncol

        # Uniform distribution over starting states
        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        def to_s(row, col):
            """Convert row col position to index"""
            return row*ncol + col


        self.shaping_coef = shaping_coef
        if self.shaping_coef > 0:
            if shaping_potential is None:
                self.shaping_potential = -create_reward_shaping_map(
                    desc, goal='G', empty=None, normalize=False)
            else:
                self.shaping_potential = np.copy(shaping_potential)


        def copmute_r(newr, newc, r, c):
            newl = desc[newr, newc]
            if self.shaping_coef > 0:
                F = (self.shaping_potential[newr, newc] -
                      self.shaping_potential[r, c])
                return float(self.base_r[newl] + self.shaping_coef * F)
            else:
                return float(self.base_r[newl])

        # Define dynamics
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        r_min = np.inf
        r_max = -np.inf

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for v in range(n_velocities):
                    for direction in range(4):

                        li = P[s][direction + 4 * v]
                        letter = desc[row, col]

                        # Terminal states
                        if letter in TERMINAL_STATES:
                            li.append((1.0, s, 0, True, {'next_state_type': letter.decode()}))
                        else:
                            if self.is_slippery:
                                for b in [(direction-1)%4, direction, (direction+1)%4]:
                                    b += v * 4
                                    rows, cols = self.inc(row, col, b)
                                    newrow, newcol = self._check_path_obstacles(rows, cols)
                                    newstate = to_s(newrow, newcol)
                                    newletter = desc[newrow, newcol]
                                    done = bytes(newletter) in TERMINAL_STATES
                                    rew = copmute_r(newrow, newcol, row, col)

                                    # Update min and max for range
                                    r_min = rew if rew < r_min else r_min
                                    r_max = rew if rew > r_max else r_max

                                    if b % 4 == direction:
                                        p = self.not_slipping_prob
                                    else:
                                        p =(1 - self.not_slipping_prob) / 2
                                    li.append((p, newstate, rew, done, {'next_state_type': newletter.decode()}))
                            else:
                                rows, cols = self.inc(row, col, direction + v*4)
                                newrow, newcol = self._check_path_obstacles(rows, cols)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in TERMINAL_STATES
                                rew = copmute_r(newrow, newcol, row, col)
                                li.append((1.0, newstate, rew, done, {'next_state_type': newletter.decode()}))

        self.reward_range = (r_min, r_max)
        super(FrozenLakeEnvCustom, self).__init__(nS, nA, P, isd, timeout)

    def _check_path_obstacles(self, rows, cols):
        is_free = np.in1d(self.desc[rows, cols], [b'S', b'F'])
        if np.all(is_free):
            return rows[-1], cols[-1]
        else:
            ind = np.argmax(~is_free)  # Index of first index with obstacle or goal
            return rows[ind], cols[ind]


    def inc(self, row, col, a):
        """Grid world dynamics"""
        direction = a % 4
        velocity = int(np.floor(a / 4)) + 1
        if direction == LEFT:
            new_col = max(col - velocity, 0)
            cols = list(range(new_col, col))[::-1]
            rows = [row] * len(cols)
        elif direction == DOWN:
            new_row = min(row + velocity, self.nrow - 1)
            rows = list(range(row + 1,  new_row + 1))
            cols = [col] * len(rows)
        elif direction == RIGHT:
            new_col = min(col + velocity, self.ncol - 1)
            cols = list(range(col + 1, new_col + 1))
            rows = [row] * len(cols)
        elif direction == UP:
            new_row = max(row - velocity, 0)
            rows = list(range(new_row, row))[::-1]
            cols = [col] * len(rows)
        if len(rows) == 0 and len(cols) == 0:  #Hitting a wall
            rows = [row]
            cols = [col]
        return (rows, cols)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    def set_state(self, s):
        self.s = s

    def get_state(self):
        return self.s

    def get_potential_value(self):
        """
        Return the value of the potential (weighted by its coeff) at the
        current state.
        """
        if self.shaping_potential is not None:
            return self.shaping_coef * self.shaping_potential[np.unravel_index(
                self.s, self.desc.shape)]
        else:
            return None


class FrozenLakeEnvCustomMap(FrozenLakeEnvCustom):
    """
    Frozen lake environment that provides the whole map as observation.
    """
    def __init__(self, desc=None, map_name="4x4", is_slippery=True,
                 not_slipping_prob=None, base_r_mapping=None, timeout=np.inf,
                 shaping_coef=0, shaping_potential=None, n_velocities=1):
        super(FrozenLakeEnvCustomMap, self).__init__(desc=desc,
                                                     map_name=map_name,
                                                     is_slippery=is_slippery,
                                                     not_slipping_prob=not_slipping_prob,
                                                     base_r_mapping=base_r_mapping,
                                                     timeout=timeout,
                                                     shaping_coef=shaping_coef,
                                                     shaping_potential=shaping_potential,
                                                     n_velocities=n_velocities)

        # Get symbols
        symbols = np.unique(self.desc)
        symbols = np.append(symbols, b'A')

        # Define numerical description of map that will be used for obs
        self.num_desc = np.zeros_like(self.desc, dtype=np.float)
        for i, symbol in enumerate(symbols):
            self.num_desc[self.desc == symbol] = i / len(symbols)

        # Redefine obs space
        n = self.observation_space.n
        shape = (int(np.sqrt(n)), int(np.sqrt(n)), 1) # Last dim for CNN
        self.observation_space = gym.spaces.Box(low=0, high=1,
                                                shape=shape, dtype=np.float)

        # Figure for rendering
        self.fig, self.ax = (None, None)

    def compute_obs(self):
        """
        Insert current position of agent in numerical description.
        """
        new_obs = np.copy(self.num_desc)
        # Agent corresponds to 1.0
        new_obs[np.unravel_index(self.s, self.num_desc.shape)] = 1.0
        return new_obs[:, :, None]

    def step(self, a):
        """
        Substitute original obs with map obs.
        """
        _, r, done, info = super(FrozenLakeEnvCustomMap, self).step(a)
        s = self.compute_obs()
        return s, r, done, info

    def reset(self):
        # Set self.s
        super(FrozenLakeEnvCustomMap, self).reset()
        return self.compute_obs()

    def render(self, mode='human', **kwargs):
        """
        Draw map with agent in it.
        """
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 10))
            self.ax = plt.gca()
        plt.cla()
        self.ax.imshow(self.compute_obs()[:, :, 0])
        plt.draw()
        plt.pause(0.01)


def fallen_in_lake(info, **kwargs):
    return info['next_state_type'] in 'H'


if __name__ == '__main__':
    env = FrozenLakeEnvCustomMap(desc=MAPS['2000lakes'])

    for i in range(100):
        a = env.action_space.sample()
        s, r, done, i = env.step(a)
        if done:
            s = env.reset()
        env.render()