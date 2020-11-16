import gym
import numpy as np

from src.envs import CMDP


__all__ = [
    'ChainEnv', 'GridEnv', 'GridEnvConstrained', 'ReachCenter']


class ChainEnv(gym.Env):
    """
    Simple chain environment with n states and two actions that deterministically go left and right. Reward is 1 at the
    right most state in the chain and -0.01 otherwise. The episode ends when the right most state is reached.
    """

    def __init__(self, n):
        self.observation_space = gym.spaces.Discrete(n)
        self.action_space = gym.spaces.Discrete(2)
        self.n = n
        self.state = 0

    def step(self, action):
        new_state = int(np.maximum(0, self.state - 1)) if action == 0 else int(np.minimum(self.n - 1, self.state + 1))
        reward = 1 if new_state == self.n - 1 else -0.01
        done = True if new_state == self.n - 1 else False
        info = {}

        self.state = new_state

        return new_state, reward, done, info

    def reset(self):
        self.state = 0
        return self.state

    def seed(self, seed):
        np.random.seed(seed)

    def get_state(self):
        return self.state

    def set_state(self, s):
        self.state = s


class GridEnv(gym.Env):
    """
    Simple nxn grid environment with four actions that deterministically go up, right, down and left. Reward is 1 at the
    right-bottom corner and -0.01 otherwise. The episode ends when the right-bottom corner state is reached.
    """

    def __init__(self, n, multi_discrete_action=True, goal_reward=1):
        self.observation_space = gym.spaces.MultiDiscrete((n, n))
        if multi_discrete_action:
            self.action_space = gym.spaces.MultiDiscrete((2, 2))

            # Action mapping
            self.u = (0, 0)
            self.r = (0, 1)
            self.d = (1, 0)
            self.l = (1, 1)

        else:
            self.action_space = gym.spaces.Discrete(4)

            # Action mapping
            self.u = 0
            self.r = 1
            self.d = 2
            self.l = 3

        self.n = n
        self.state = (0, 0)
        self.goal_reward = goal_reward
        self.reward_range = (-0.01, goal_reward)

    def step(self, action):
        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            action = tuple(action)

        if action == self.u:
            new_x = self.state[0]
            new_y = int(np.minimum(self.n - 1, self.state[1] + 1))

        elif action == self.r:
            new_x = int(np.minimum(self.n - 1, self.state[0] + 1))
            new_y = self.state[1]

        elif action == self.d:
            new_x = self.state[0]
            new_y = int(np.maximum(0, self.state[1] - 1))

        else:
            new_x = int(np.maximum(0, self.state[0] - 1))
            new_y = self.state[1]
        new_state = (new_x, new_y)

        reward = self.goal_reward if new_state == (self.n - 1, self.n - 1) else -0.01
        done = True if new_state == (self.n - 1, self.n - 1) else False
        info = {}

        self.state = new_state

        return new_state, reward, done, info

    def reset(self):
        self.state = (0, 0)
        return self.state

    def seed(self, seed):
        np.random.seed(seed)

    def get_state(self):
        return self.state

    def set_state(self, s):
        self.state = s


class GridEnvConstrained(CMDP):
    def __init__(self, n, constraint, multi_discrete_action=True,
                 goal_reward=1, **kwargs):
        env = GridEnv(n, multi_discrete_action, goal_reward)
        super().__init__(env, constraint, **kwargs)

    def step(self, action):
        obs, r, g, done, info = super().step(action)
        done |= g[0] > 0
        return obs, r, g, done, info


class ReachCenter(gym.Env):
    """
    Two dimensional grid with four actions. The goal is to reach the center
    of the grid. Every action that takes us closer to the center gets +1,
    -1 to get further. Moreover there is an additionaly 0.1 penalty for moving
    left/right so that the optimal policy has one single action
    """

    def __init__(self, n):
        assert np.mod(n, 2) != 0, 'number of tiles must be odd'
        self.observation_space = gym.spaces.MultiDiscrete((n, n))
        self.action_space = gym.spaces.MultiDiscrete((2, 2))
        self.n = n
        self.initial_states = [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]
        self.state = self.initial_states[np.random.choice(len(
            self.initial_states))]

        self.target = (int(np.floor(n / 2)), int(np.floor(n / 2)))

        # Action mapping
        self.u = (0, 0)
        self.r = (0, 1)
        self.d = (1, 0)
        self.l = (1, 1)

    def step(self, action):
        # Up
        if action[0] == 0 and action[1] == 0:
            new_x = self.state[0]
            new_y = int(np.minimum(self.n - 1, self.state[1] + 1))
        # Right
        elif action[0] == 0 and action[1] == 1:
            new_x = int(np.minimum(self.n - 1, self.state[0] + 1))
            new_y = self.state[1]
        # Down
        elif action[0] == 1 and action[1] == 0:
            new_x = self.state[0]
            new_y = int(np.maximum(0, self.state[1] - 1))
        # Left
        else:
            new_x = int(np.maximum(0, self.state[0] - 1))
            new_y = self.state[1]
        new_state = (new_x, new_y)
        new_dist = np.linalg.norm([new_state[i] - self.target[i] for i in
                                   range(2)], ord=1)
        old_dist = np.linalg.norm([self.state[i] - self.target[i] for i in
                                   range(2)], ord=1)
        reward = 1 if new_dist < old_dist else -1
        if tuple(action) in [self.l, self.r]:
            reward -= 0.1
        done = new_state == self.target
        info = {}

        self.state = new_state

        return new_state, reward, done, info

    def reset(self):
        self.state = self.initial_states[np.random.choice(len(
            self.initial_states))]
        return self.state

    def seed(self, seed):
        np.random.seed(seed)

    def get_state(self):
        return self.state

    def set_state(self, s):
        self.state = s


class DummyTeacherEnvSmallFrozen(gym.Env):
    """
    A  quick environment that allows us to reproduce the behavior of the
    teacher env when supervising students in the small lake env.

    In the small frozen lake, there are two actions: action 0 helps solving
    the exploration problem but differs from the final env (therefore the
    amount of reward we can accumulate training with action 0 and testing in
    the final env is limited); action 0 is the final env. If proposed
    directly, it is not solvable due to exploration problems. If proposed
    after some training has happened with action 0, then it leads to higher
    rewards.

    This is recreated by a chain, where the state is the node number in the
    chain and the observation is the percentage of completion of the chain
    from left to right. Action 0 pushes us always to the middle of the
    chain while action 1 always to the closest extreme. The reward
    is the difference in state.
    """

    def __init__(self, n, p, sigma_r, timeout=None):
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.action_space = gym.spaces.Discrete(2)
        self.n = n  # Length of the chain
        self.p = p  # Probability of action having desired consequences
        self.sigma_r = sigma_r # Gaussian noise on the reward
        self.center = int(self.n / 2)
        self.state = 0
        self.n_steps = 0
        self.timeout = timeout if timeout is not None else n * 2

    def step(self, action):
        # First half of the chain
        if self.state <= self.center:
            if action == 0:
                new_state = self.state + 1 if np.random.rand(1) <= self.p \
                    else self.state
            elif action == 1:
                new_state = self.state - 1 if np.random.rand(1) <= self.p \
                    else self.state
                new_state = np.maximum(new_state, 0)
        else:
            if action == 1:
                new_state = self.state + 1 if np.random.rand(1) <= self.p  \
                    else self.state
                new_state = np.minimum(new_state, self.n)
            elif action == 0:
                new_state = self.state - 1 if np.random.rand(1) <= self.p \
                    else  self.state
        reward = new_state - self.state + self.sigma_r * np.random.randn(1)[0]
        self.state = new_state
        done = True if self.n_steps > self.timeout else False
        info = {}
        obs = [new_state / self.n]
        self.n_steps += 1

        return obs, reward, done, info

    def reset(self):
        self.n_steps = 0
        self.state = 0
        return [self.state]

    def seed(self, seed):
        np.random.seed(seed)

    def get_state(self):
        return self.state

    def set_state(self, s):
        self.state = s

if __name__ == '__main__':
    env = DummyTeacherEnvSmallFrozen(10, 0.99, 0.01, 10)
    actions = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    for a in actions:
        obs, r, done, info = env.step(a)
        print(r, env.state)