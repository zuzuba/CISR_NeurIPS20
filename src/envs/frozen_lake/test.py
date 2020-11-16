import gym
import numpy as np
import unittest
from numpy.testing import *

import src.envs.frozen_lake.flake_constants
from src.envs.frozen_lake.frozen_maps import MAPS
from src.envs import *
from src.envs.frozen_lake.utils import OptimalAgent, \
    create_reward_shaping_map, value_iteration
import src.envs.frozen_lake.frozen_lake_custom as lake_custom


class TestCustomFrozenEnvs(unittest.TestCase):
    """
    Test for different versions of the custom frozen environment.

    In particular, we test the base env, the slippery env, the reward
    shaping, the constrained env and the Lagrangian MDP derived from the CMDP.
    Almost all tests consist of taking a predefined set of actions and
    checking that observations, rewards and so on are as expected.
    """
    def setUp(self):
        # Base environments
        self.env = FrozenLakeEnvCustom(is_slippery=False, map_name='test')
        self.slippery_env = FrozenLakeEnvCustom(is_slippery=True,
                                                map_name='test')
        self.env_r_shaping = FrozenLakeEnvCustom(is_slippery=False,
                                                 map_name='test',
                                                 shaping_coef=1)

        # Constrained MDP
        def constraint(info, **kwargs):
            return {'g': fallen_in_lake(info, **kwargs)}

        self.cenv = CMDP(FrozenLakeEnvCustom(is_slippery=False,
                                             map_name='test'), constraint)

        # Lagrangian MDP and monitor
        self.lenv = LagrangianMDP(self.cenv, 0.5)
        self.monitor = LagrangianMDPMonitor(self.lenv)

        # Actions and reward mappings
        self.L = src.envs.frozen_lake.flake_constants.LEFT
        self.U = src.envs.frozen_lake.flake_constants.UP
        self.R = src.envs.frozen_lake.flake_constants.RIGHT
        self.D = src.envs.frozen_lake.flake_constants.DOWN

        self.r = {k.decode(): v for k, v in src.envs.frozen_lake.flake_constants.REWARD_MAPPING.items()}

    def test_standard(self):
        # To danger
        actions = [self.D] * 4 + [self.R] * 2
        s = np.zeros(len(actions), dtype=int)
        r = np.zeros(len(actions), dtype=float)
        d = np.zeros(len(actions), dtype=bool)
        tiles = np.zeros(len(actions), dtype=str)

        self.env.reset()
        for i, a in enumerate(actions):
            s[i], r[i], d[i], info = self.env.step(a)
            tiles[i] = info['next_state_type']

        assert_array_equal(s, [61, 86, 111, 136, 137, 138])
        assert_array_equal(r, [self.r['F']] * 5 + [self.r['H']])
        assert_array_equal(d, [False] * 5 + [True])
        assert_array_equal( tiles, ['F'] * 5 + ['H'])

        # To goal
        actions = [self.D] * 10 + [self.R] * 4
        s = np.zeros(len(actions), dtype=int)
        r = np.zeros(len(actions), dtype=float)
        d = np.zeros(len(actions), dtype=bool)
        tiles = np.zeros(len(actions), dtype=str)

        self.env.reset()
        for i, a in enumerate(actions):
            s[i], r[i], d[i], info = self.env.step(a)
            tiles[i] = info['next_state_type']

        assert_array_equal(s, [61, 86, 111, 136, 161, 186, 211, 236, 261, 286, 287, 288, 289, 290])
        assert_array_equal(r, [self.r['F']] * 13 + [self.r['G']])
        assert_array_equal(d, [False] * 13 + [True])
        assert_array_equal(tiles, ['F'] * 13 + ['G'])

    def test_r_shaping(self):
        # Test directions
        actions = [self.D, self.U, self.R, self.L]
        targets = [self.r['F'] + 1 * self.env_r_shaping.shaping_coef,
                   self.r['F'] - 1 * self.env_r_shaping.shaping_coef,
                   self.r['F'] + 1 * self.env_r_shaping.shaping_coef,
                   self.r['F'] - 1 * self.env_r_shaping.shaping_coef]
        for a, t in zip(actions, targets):
            self.env_r_shaping.reset()
            _, r, _, _  = self.env_r_shaping.step(a)
            assert_equal(r, t)

        # To danger
        actions = [self.D] * 4 + [self.R] * 2
        r = np.zeros(len(actions), dtype=float)

        self.env_r_shaping.reset()
        for i, a in enumerate(actions):
            _, r[i], _, info = self.env_r_shaping.step(a)
        target_r = [self.r['F'] + 1 * self.env_r_shaping.shaping_coef] * 5 + \
                   [self.r['H'] + 1 * self.env_r_shaping.shaping_coef]
        # target_r = np.append(self.env_r_shaping.r_shaping_coeff * (1 - np.arange(13, 8, -1) / 26), [self.r['H']])
        assert_array_almost_equal(r, target_r, decimal=4)

        # To goal
        actions = [self.D] * 10 + [self.R] * 4
        r = np.zeros(len(actions), dtype=float)

        self.env_r_shaping.reset()
        for i, a in enumerate(actions):
            _, r[i], _, _ = self.env_r_shaping.step(a)
        target_r = [self.r['F'] + 1 * self.env_r_shaping.shaping_coef] * 13 + \
                   [self.r['G'] + 1 * self.env_r_shaping.shaping_coef]
        assert_array_almost_equal(r, target_r)

    def test_env_with_velocities(self):
        # Action format: direction + 4 * (velocity - 1)
        env = FrozenLakeEnvCustom(is_slippery=False, map_name='test', shaping_coef=0, n_velocities=3)

        # To goal
        actions = [self.D + 4 * 2, self.D + 4 * 2, self.D + 4 * 1, self.D + 4 * 1, self.R + 4 * 1, self.R + 4 * 0, self.R + 4 * 0]
        s = np.zeros(len(actions), dtype=int)
        r = np.zeros(len(actions), dtype=float)
        d = np.zeros(len(actions), dtype=bool)
        tiles = np.zeros(len(actions), dtype=str)

        env.reset()
        for i, a in enumerate(actions):
            s[i], r[i], d[i], info = env.step(a)
            tiles[i] = info['next_state_type']

        assert_array_equal(s, [111, 186, 236, 286, 288, 289, 290])
        assert_array_equal(r, [self.r['F']] * 6 + [self.r['G']])
        assert_array_equal(d, [False] * 6 + [True])
        assert_array_equal(tiles, ['F'] * 6 + ['G'])

        # To danger
        actions = [self.D + 4 * 2, self.D + 4 * 2, self.R + 4 * 2]

        s = np.zeros(len(actions), dtype=int)
        r = np.zeros(len(actions), dtype=float)
        d = np.zeros(len(actions), dtype=bool)
        tiles = np.zeros(len(actions), dtype=str)

        env.reset()
        for i, a in enumerate(actions):
            s[i], r[i], d[i], info = env.step(a)
            tiles[i] = info['next_state_type']

        assert_array_equal(s, [111, 186, 188])
        assert_array_equal(r, [self.r['F']] * 2 + [self.r['H']])
        assert_array_equal(d, [False] * 2 + [True])
        assert_array_equal(tiles, ['F'] * 2 + ['H'])

        # Try to jump obstacle (should not manage cause velocities do not actually allow you to skip tiles)
        # To danger
        actions = [self.D + 4 * 2, self.D + 4 * 2, self.L + 4 * 0, self.L + 4 * 2]

        s = np.zeros(len(actions), dtype=int)
        r = np.zeros(len(actions), dtype=float)
        d = np.zeros(len(actions), dtype=bool)
        tiles = np.zeros(len(actions), dtype=str)

        env.reset()
        for i, a in enumerate(actions):
            s[i], r[i], d[i], info = env.step(a)
            tiles[i] = info['next_state_type']

        assert_array_equal(s, [111, 186, 185, 184])
        assert_array_equal(r, [self.r['F']] * 3 + [self.r['H']])
        assert_array_equal(d, [False] * 3 + [True])
        assert_array_equal(tiles, ['F'] * 3 + ['H'])

    def test_slippery_env(self):
        np.random.seed(0)
        n_trials = 500
        next_s = np.zeros(n_trials)
        for i in range(n_trials):
            self.slippery_env.reset()
            next_s[i], _, _, _ = self.slippery_env.step(self.R)
        assert np.abs(np.sum(next_s == 37) / n_trials - self.env.not_slipping_prob) < 0.05
        assert np.abs(np.sum(next_s == 61) / n_trials - (1 - self.env.not_slipping_prob) / 2) < 0.05
        assert np.abs(np.sum(next_s == 11) / n_trials - (1 - self.env.not_slipping_prob) / 2) < 0.05

    def test_cenv_no_teacher(self):
        # To danger
        actions = [self.D, self.D, self.D, self.D, self.R, self.R]
        reward = np.zeros_like(actions, dtype=float)
        constraints = np.zeros_like(actions, dtype=float)

        self.cenv.reset()
        for i, a in enumerate(actions):
            obs, reward[i], g_tmp, done, info = self.cenv.step(a)
            constraints[i] = g_tmp[0]

        assert_array_almost_equal(reward, np.array([self.r['F']] * 5 + [self.r['H']]))
        assert_array_equal(constraints, np.array([0, 0, 0, 0, 0, 1]))

        # To goal
        actions = [self.D] * 10 + [self.R] * 4
        reward = np.zeros_like(actions, dtype=float)
        constraints = np.zeros_like(actions, dtype=float)

        self.cenv.reset()
        for i, a in enumerate(actions):
            obs, reward[i], g_tmp, done, info = self.cenv.step(a)
            constraints[i] = g_tmp[0]

        assert_array_equal(reward, np.array([self.r['F']] * 13 + [self.r['G']]))
        assert_array_equal(constraints, np.array([0] * 14))

    def test_monitor(self):
        """Test for one episode leading to danger and one leading to goal"""
        scenario_actions = [[self.D] * 4 + [self.R] * 2,  # To danger
                            [self.D] * 10 + [self.R] * 4]  # To goal

        lagangian_r = np.zeros(len(scenario_actions))

        for i, actions in enumerate(scenario_actions):
            self.monitor.reset()
            for a in actions:
                _, r, done, _ = self.monitor.step(a)
                lagangian_r[i] += r

        target_episodes_r = [self.r['F'] * 5 + self.r['H'], self.r['F'] * 13 + self.r['G']]
        target_episodes_g = [[1], [0]]
        target_episodes_l = [6, 14]
        target_episodes_lagrangian_r = [self.r['F'] * 5 + self.r['H'] - self.monitor.lam, self.r['F'] * 13 + self.r['G']]

        assert_array_equal(self.monitor.get_episode_rewards(), target_episodes_r)
        assert_array_equal(self.monitor.get_episode_constraints(), target_episodes_g)
        assert_array_equal(self.monitor.get_episode_lengths(), target_episodes_l)
        assert_array_equal(lagangian_r, target_episodes_lagrangian_r)

    def test_timeout(self):
        env = FrozenLakeEnvCustom(is_slippery=False, map_name='test',
                                  timeout=10)
        env.reset()
        for i in range(15):
            _, _, done, _ = env.step(self.U)
            if i == 10:
                assert done
            else:
                assert not done
            if done:
                env.reset()

    def test_custom_reward(self):
        r_map = {b'G': -6, b'S': -0.01, b'F': -1, b'H': 5}
        env = FrozenLakeEnvCustom(is_slippery=False,
                                  map_name='test',
                                  base_r_mapping=r_map)
        env.reset()
        actions = [self.D] * 4 + [self.R] * 2
        target_r_list = [-1] * 5 + [5]
        for a, target_r in zip(actions, target_r_list):
            _, r, _, _ = env.step(a)
            assert r == target_r

    def test_custom_potential(self):
        # Set to zero so only the potential matters
        r_map = {b'G': 0, b'S': 0, b'F': 0, b'H': 0}
        shape = np.array(MAPS['test'], dtype='c').shape
        potential = np.arange(np.prod(shape)).reshape(shape)
        env = FrozenLakeEnvCustom(is_slippery=False,
                                  map_name='test',
                                  base_r_mapping=r_map,
                                  shaping_potential=potential,
                                  shaping_coef=1.)
        actions = [self.U, self.R, self.D, self.L]
        target_r_list = [-25, 1, 25, -1]
        for a, target_r in zip(actions, target_r_list):
            env.reset()
            _, r, _, _ = env.step(a)
            assert r == target_r

    def test_constratin_values(self):

        # Constrained MDP
        def constraint(info, **kwargs):
            return {'g': fallen_in_lake(info, **kwargs)}

        # With avg constraint
        cenv = CMDP(FrozenLakeEnvCustom(is_slippery=False,
                                        map_name='test'), constraint,
                    avg_constraint=True, constraints_values=[0.1])

        actions_list = [[self.D] * 4 + [self.R] * 2] + \
                       [[self.D] * 10 + [self.R] * 4] + \
                       [[self.R] * 6 + [self.D] * 5 + [self.L]]

        g_final = []

        for actions in actions_list:
            cenv.reset()
            g_ep = np.zeros(1)
            for i, a in enumerate(actions):
                _, _, g, _, _ = cenv.step(a)
                g_ep += g
            g_final.append(g_ep)
        assert_almost_equal(g_final, [[1./6-0.1], [0.-0.1], [1./12-0.1]])

        # With cumulative constraint
        cenv = CMDP(FrozenLakeEnvCustom(is_slippery=False,
                                        map_name='test'), constraint,
                    avg_constraint=False, constraints_values=[0.5])

        g_final = []

        for actions in actions_list:
            cenv.reset()
            g_ep = np.zeros(1)
            for i, a in enumerate(actions):
                _, _, g, _, _ = cenv.step(a)
                g_ep += g
            g_final.append(g_ep)
        assert_almost_equal(g_final,
                            [[0.5], [-0.5], [0.5]])


class TestOptimalAgent(unittest.TestCase):
    def test_agent(self):
        env = FrozenLakeEnvCustomMap(is_slippery=False,
                                     map_name='4x4',
                                     base_r_mapping={b'G': 6,
                                                     b'S': -0.01,
                                                     b'F': -0.01,
                                                     b'H': -6})
        path = np.array([[0, i] for i in range(3)] +
                        [[i, 2] for i in range(1, 4)] +
                        [[3, 3]])
        agent = OptimalAgent(env.desc, path=path, reward_map=env.base_r)
        l = src.envs.frozen_lake.flake_constants.LEFT
        u = src.envs.frozen_lake.flake_constants.UP
        r = src.envs.frozen_lake.flake_constants.RIGHT
        d = src.envs.frozen_lake.flake_constants.DOWN

        actions = {(0, 0): r, (0, 1): r, (0, 2): d, (0, 3): l,
                   (1, 0): u, (1, 2): d,
                   (2, 1): r, (2, 2): d,
                   (3, 1): r, (3, 2): r}
        for k, v in actions.items():
            assert agent.policy[k[0], k[1]] == v

        # Remove state that have more than one optimal action and modify
        # optimal action for states where it changed
        del actions[(0, 0)]
        del actions[(2, 1)]
        actions[(1, 0)] = d
        agent = OptimalAgent(env.desc, reward_map=env.base_r)
        for k, v in actions.items():
            assert agent.policy[k[0], k[1]] == v

    def test_policy_iteration(self):
        """
        Test policy iteration with stochastic small env.
        """
        simple_map = ['SFH', 'HFH', 'HFG']
        r = np.array([[-1, -1, 0], [0, -1, 0], [0, -1, 10]])
        terminal_state_mask = np.array([[0, 0, 1], [1, 0, 1], [1, 0, 1]],
                                       dtype=bool)
        policy, value = value_iteration(r, terminal_state_mask,
                                        not_slipping_prob=0.8, eps=0)
        r, l, u, d = [src.envs.frozen_lake.flake_constants.RIGHT,
                      src.envs.frozen_lake.flake_constants.LEFT,
                      src.envs.frozen_lake.flake_constants.UP,
                      src.envs.frozen_lake.flake_constants.DOWN]
        target_policy = [[r, d, l], [r, d, d], [r, r, policy[2, 2]]]
        target_value = [[2.278405710886378, 3.813206424747175, 0],
                        [0, 5.731707317073171, 0],
                        [0, 8.414634146341465, 10]]
        assert_almost_equal(policy, target_policy)
        assert_almost_equal(value, target_value)


class TestRewardMapCreation(unittest.TestCase):
    def test(self):
        base_map = ['   G ',
                    ' G   ',
                    '     ',
                    'G    ']
        target = np.array([[2, 1, 1, 0, 1],
                           [1, 0, 1, 1, 2],
                           [1, 1, 2, 2, 3],
                           [0, 1, 2, 3, 4]])
        assert_array_equal(target, create_reward_shaping_map(base_map, empty=' ', normalize=False))
        target = target.astype(float)
        assert_array_almost_equal(target / np.max(target), create_reward_shaping_map(base_map, empty=' ', normalize=True))

        # With bytes map
        base_map = np.array(base_map, dtype='c')
        assert_array_equal(target, create_reward_shaping_map(base_map, empty=' ', normalize=False))
        target = target.astype(float)
        assert_array_almost_equal(target / np.max(target), create_reward_shaping_map(base_map, empty=' ', normalize=True))


if __name__ == '__main__':
    unittest.main()

