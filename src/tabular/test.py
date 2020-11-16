import numpy as np
import unittest
from numpy.testing import *


from src.tabular.policies import TabularQPolicy, Qdict2array
from src.tabular.TD import QLearning
from src.envs.dummy_envs import *


class TestQLearning(unittest.TestCase):
    def setUp(self):
        self.env = ChainEnv(6)
        self.env_multi = GridEnv(n=3, multi_discrete_action=True,
                                 goal_reward=1)
        self.model = QLearning(TabularQPolicy, self.env)
        self.model_multi = QLearning(TabularQPolicy, self.env_multi)

    def test_Q_learning(self):
        self.model.learn(2000)
        targetQ = np.array([[.95, .96],
                            [.95, .97],
                            [.96, .98],
                            [.97, .99],
                            [.98, 1.0],
                            [0.0, 0.0]])

        assert_array_almost_equal(targetQ, Qdict2array(self.model.get_parameters()['Q']))

    def test_predict(self):
        # Check that predicts does not use softmax but greedy policy by default
        self.model.learn(2000)
        targetQ = np.array([[.95, .96],
                            [.95, .97],
                            [.96, .98],
                            [.97, .99],
                            [.98, 1.0],
                            [0.0, 0.0]])
        for i in range(100):
            assert_equal(self.model.predict(0)[0], 1)

    def test_Q_learning_multi(self):
        """
        Test Q learning with multi-discrete action and state space.
        """
        self.model_multi.learn(8000)

        targetQ = np.array([[.97, .97, .96, .96],
                            [.98, .98, .96, .97],
                            [.98, .99, .97, .98],
                            [.98, .98, .97, .96],
                            [.99, .99, .97, .97],
                            [.99, 1.0, .98, .98],
                            [.99, .98, .98, .97],
                            [1.0, .99, .98, .98],
                            [0.0, 0.0, 0.0, 0.0]])
        assert_array_almost_equal(targetQ, Qdict2array(self.model_multi.get_parameters()['Q']), decimal=4)


class TestTabularQPolicy(unittest.TestCase):
    def setUp(self):
        self.env = GridEnv(n=3, multi_discrete_action=True,
                                 goal_reward=1)
        self.policy = TabularQPolicy(self.env.observation_space, self.env.action_space)

    def test(self):
        # Update the Q values
        s = [(0, 0), (0, 0)]
        a = [(0, 1), (0, 0)]
        vals = [1, 2]
        self.policy.update_Q(s, a, vals)

        # Check the probability distribution and the sampling are right
        denominator = 2 * np.exp(0) + np.exp(1) + np.exp(2)
        target = [np.exp(2) / denominator, np.exp(1) / denominator, 1/denominator, 1/denominator]
        assert_almost_equal(target, self.policy.proba_step((0, 0)))

        counter = np.array([0] * 4)
        for i in range(10000):
            a = self.policy.step((0, 0), deterministic=False)
            for i, action_tuple in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                if tuple(a) == action_tuple:
                    counter[i] += 1
        assert_allclose(target, counter / np.sum(counter), atol=0.01)

        # Change temperature and repeat the test
        self.policy.temperature = 10
        denominator = 2 * np.exp(0) + np.exp(0.1) + np.exp(0.2)
        target = [np.exp(0.2) / denominator, np.exp(0.1) / denominator, 1 / denominator, 1 / denominator]
        assert_almost_equal(target, self.policy.proba_step((0, 0)))

        counter = np.array([0] * 4)
        for i in range(10000):
            a = self.policy.step((0, 0), deterministic=False)
            for i, action_tuple in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                if tuple(a) == action_tuple:
                    counter[i] += 1
        assert_allclose(target, counter / np.sum(counter), atol=0.01)


if __name__ == '__main__':
    unittest.main()