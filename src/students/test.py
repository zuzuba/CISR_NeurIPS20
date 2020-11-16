import numpy as np
import unittest
from numpy.testing import *
import re
import tensorflow as tf

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc


from src.students import *
from src.tabular import QLearning, TabularQPolicy
from src.online_learning import ExponetiatedGradient
from src.envs import GridEnvConstrained, FrozenLakeEnvCustom, \
    FrozenLakeEnvCustomMap, CMDP


class TestIdentityTransfer(unittest.TestCase):
    def setUp(self):
        self.n = 4

        def constraint(observation, **kwargs):
            if observation[0] != 0 and observation[0] != self.n - 1 \
                    and observation[1] != 0 and observation[1] != self.n - 1:
                return {'g': 1}
            else:
                return {'g': 0}

        self.env = GridEnvConstrained(self.n, constraint,
                                      multi_discrete_action=True,
                                      goal_reward=0)
        self.S_identity = LagrangianStudent(env=None,
                                            br_algo=QLearning,
                                            online_algo=ExponetiatedGradient,
                                            br_kwargs={'policy': TabularQPolicy, 'lr_final': 0.1,
                                                       'temp_fraction': 0.8},
                                            online_kwargs={'B': 10, 'eta': 0.5},
                                            curriculum_transfer=identity_transfer)

    def test(self):
        self.S_identity.set_env(self.env)
        self.S_identity.learn(100)
        old_params = self.S_identity.get_params()
        self.S_identity.set_env(self.env)
        new_params = self.S_identity.get_params()

        assert_array_equal(old_params['multipliers'], new_params['multipliers'])

        for s, q_val in new_params['Q'].items():
            q_val_old = old_params['Q'][s]
            assert_array_equal(q_val, q_val_old)

        for s, pi_s in new_params['pi'].items():
            pi_s_old = old_params['pi'][s]
            assert_array_equal(pi_s, pi_s_old)

def my_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=2, filter_size=2,
                         stride=1,**kwargs))
    layer_3 = conv_to_fc(layer_1)
    return activ(linear(layer_3, 'fc1', n_hidden=64, init_scale=np.sqrt(2)))


class TestResetValueTransfer(unittest.TestCase):
    def setUp(self):
        def make_env():
            """
            Env with position observation.
            """
            base_env = FrozenLakeEnvCustom(map_name='4x4',
                                           not_slipping_prob=0.8)
            def constraint(**kwargs):
                return {'g': False}

            return CMDP(base_env, constraint)

        def make_cnn_env():
            """
            Env with full map observation (for CNN).
            """
            base_env = FrozenLakeEnvCustomMap(map_name='4x4',
                                              not_slipping_prob=0.8)

            def constraint(**kwargs):
                return {'g': False}

            return CMDP(base_env, constraint)
        
        def check_params(old, new):
            """
            Check that all parameters are the same except those that involve
            exclusively the value function.
            """
            for k in old.keys():
                old_val = old[k]
                new_val = new[k]
                if re.match('.*vf.*', k):
                    assert_raises(AssertionError, assert_array_equal,
                                  old_val, new_val)
                else:
                    assert_array_equal(old_val, new_val)

        self.make_env = make_env
        self.make_cnn_env = make_cnn_env
        self.assert_correct_params = check_params

    def test(self):
        """
        Define student with both MLP and CNN, learn from env and reset them.
        Check that after the reset everything is the same except for value
        function.
        """

        policies = [MlpPolicy, CnnPolicy]
        policy_kwargs = [{'net_arch': [32, dict(vf=[16], pi=[2])]},
                         {'cnn_extractor': my_cnn}]
        envs = [self.make_env, self.make_cnn_env]

        for env, p, p_kwargs in zip(envs, policies, policy_kwargs):
            print(p_kwargs)
            student = LagrangianStudent(env=env,
                                        br_algo=PPO2,
                                        online_algo=ExponetiatedGradient,
                                        br_kwargs={'policy': p,
                                                   'policy_kwargs': p_kwargs},
                                        online_kwargs={'B': 10, 'eta': 0.5},
                                        curriculum_transfer=reset_ppo_vf,
                                        br_uses_vec_env=True,
                                        use_sub_proc_env=False,
                                        lagrangian_ronuds=1)
            student.learn(400) # Learning is necessary otherwise the bias stays 0
            old_params = student.get_br_params()
            student.set_env(env)
            new_params = student.get_br_params()
            self.assert_correct_params(old_params, new_params)


if __name__ == '__main__':
    unittest.main()