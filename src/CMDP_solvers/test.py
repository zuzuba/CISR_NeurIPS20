import numpy as np
import unittest
from numpy.testing import *

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

from src.CMDP_solvers import LagrangianCMDPSolver
from src.tabular import QLearning, TabularQPolicy
from src.online_learning import ExponetiatedGradient
from src.envs.dummy_envs import *

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class TestLagrangianSolver(unittest.TestCase):
    def setUp(self):
        """
        Setup basic solver with tabular best response algorithm and basic env.
        """
        # Environment definition
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

        # Tabular solver definition
        self.solver = LagrangianCMDPSolver(env=None,
                                           br_algo=QLearning,
                                           online_algo=ExponetiatedGradient,
                                           br_kwargs={'policy': TabularQPolicy, 'lr_final': 0.1, 'temp_fraction': 0.8},
                                           online_kwargs={'B': 10, 'eta': 0.5})

    def test_solve(self):
        """
        Test solution with tabular best response algorithm.
        """

        with self.assertRaises(ValueError):
            self.solver.learn(100)

        self.solver.set_env(self.env)

        self.solver.learn(5000, log=True)
        pi = self.solver.br.get_parameters()['pi']

        # Test policy
        u, r, l, d, = (self.env.u, self.env.r, self.env.l, self.env.d)
        target_policy = {(0, 1): u, (0, 2): u, (0, 3): r,
                         (1, 3): r, (2, 3): r,
                         (1, 0): r, (2, 0): r, (3, 0): u,
                         (3, 1): u, (3, 2): u}

        for s, a_prob in pi.items():
            if s in target_policy.keys():
                learned_a = np.unravel_index(np.argmax(a_prob), (2, 2))
                target_a = target_policy[s]
                assert_array_equal(learned_a, target_a)

        # Test value function
        target_V = {(0, 0): -.05, (0, 1): -.04, (0, 2): -.03, (0, 3): -.02,
                    (1, 3): -.01, (2, 3): -.00,
                    (1, 0): -.04, (2, 0): -.03, (3, 0): -.02,
                    (3, 1): -.01, (3, 2): -.00,
                    }

        Q = self.solver.br.get_parameters()['Q']
        for s, q_val in Q.items():
            if s in target_V:
                learned_V = np.max(q_val)
                assert_almost_equal(target_V[s], learned_V, decimal=4)

    def test_solve_vec_enc(self):
        """
        Test solution with br algorithm that requires vectorized env.

        Since PPO is a local solver, we modify the set up slightly so that
        identifying the optimal policy becomes easier: We allow for 0.05
        constraint violation, we set the upper bound of the multipliers to 2
        and we give a reward of 1 for reaching the goal.
        """
        # Env specification (for vectorized case, we specify an env
        # generating callable)
        n = 4

        def constraint(observation, **kwargs):
            if (observation[0] == 1 or observation[0] == 1) and \
               (observation[1] == 2 or observation[1] == 2):
                return {'g': 1}
            else:
                return {'g': 0}

        def env_generator():
            return GridEnvConstrained(
                n, constraint, multi_discrete_action=False,
                goal_reward=1, constraints_values=[0.05])

        # Specify PPO kwargs
        br_kwargs = dict(policy=MlpPolicy, ent_coef=0.008,
                         learning_rate=2e-4,
                         vf_coef=0.5,
                         max_grad_norm=0.98,
                         lam=0.95,
                         cliprange=0.1,
                         noptepochs=30,
                         policy_kwargs={'net_arch': [16, 16]})

        # Define CMDP solver
        ppo_solver = LagrangianCMDPSolver(
            env=None,
            lagrangian_ronuds=10,
            br_algo=PPO2,
            online_algo=ExponetiatedGradient,
            br_kwargs=br_kwargs,
            online_kwargs={'B': 2, 'eta': 0.5},
            br_uses_vec_env=True,
            use_sub_proc_env=False,
            n_envs=2,
            log_training=True)

        ppo_solver.set_env(env_generator)

        ppo_solver.learn(6000)

        # Testing with number of successes rather than optimal policy for
        # every state (easier with function approximation)
        new_env = env_generator()
        obs = new_env.reset()
        n_episodes = 100
        successes = 0
        for i in range(n_episodes):
            for j in range(100):
                a, _ = ppo_solver.predict(obs, deterministic=False)
                obs, r, g, done, info = new_env.step(a)

                if done:
                    successes += int(obs == (n - 1, n - 1))
                    obs = new_env.reset()
                    break
        assert successes / n_episodes >= 0.8


    def test_update_multipliers(self):
        """
        Test when changing environments whether the multipliers are set correctly
        Returns
        -------
        """
        self.solver.set_env(self.env)
        _, _, old_w = self.solver.learn(100, log=True)
        self.solver.set_env(self.env, keep_multipliers=True)
        assert_array_equal(old_w, self.solver.online.w)

        self.solver.set_env(self.env)
        assert_array_equal(self.solver.online.w, [5, 5])

    def test_set_env(self):
        """
        Test that setting the environment does not raise pickling error when
        using multiprocessing vectorized environments.
        """
        n = 4

        def constraint(observation, **kwargs):
            if (observation[0] == 1 or observation[0] == 1) and \
                    (observation[1] == 2 or observation[1] == 2):
                return {'g': 1}
            else:
                return {'g': 0}

        # Due to serialization, this env cannot be defined in the setup
        def env_generator():
            return GridEnvConstrained(
                n, constraint, multi_discrete_action=False,
                goal_reward=1, constraints_values=[0.05])

        # Specify PPO kwargs
        br_kwargs = dict(policy=MlpPolicy)
        ppo_solver = LagrangianCMDPSolver(
            env=env_generator,
            br_algo=PPO2,
            online_algo=ExponetiatedGradient,
            br_kwargs=br_kwargs,
            br_uses_vec_env=True,
            use_sub_proc_env=True,
            n_envs=1,)

        ppo_solver.set_env(env_generator)


if __name__ == '__main__':
    unittest.main()
