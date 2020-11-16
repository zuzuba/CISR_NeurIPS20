import numpy as np
import unittest
from numpy.testing import *

from src.teacher import *
from src.teacher.common import set_dict_values
from src.envs import GridEnv, CMDP, GridEnvConstrained, FrozenLakeEnvCustom
from src.envs.frozen_lake.flake_constants import DOWN, RIGHT
from src.tabular import QLearning, TabularQPolicy
from src.online_learning import ExponetiatedGradient
from src.students import LagrangianStudent, identity_transfer


class TestTeacherIntervention(unittest.TestCase):
    def setUp(self):
        self.base_env = GridEnv(n=4, multi_discrete_action=True,
                                 goal_reward=1)

        def constraint(observation, **kwargs):
            return {'g': float(observation == (1, 1))}

        self.cenv = CMDP(self.base_env, constraint)

        def intervention_condition1(observation, **kwargs):
            return float(observation == (2, 2))

        def intervention_condition2(observation, **kwargs):
            return float(observation == (2, 3))

        self.tenv_single = TeacherIntervention(self.cenv, intervention_condition1)
        self.tenv_multi = TeacherIntervention(self.cenv, [intervention_condition1, intervention_condition2])
        self.tenv_single_buff = TeacherIntervention(self.cenv, intervention_condition1, buffer_size=1)
        self.tenv_multi_buff = TeacherIntervention(self.cenv, [intervention_condition1, intervention_condition2],
                                                   buffer_size=1)
    def test_interventions(self):
        actions_to_failure = [self.base_env.u, self.base_env.r]
        actions_to_t1 = [self.base_env.r, self.base_env.r, self.base_env.u, self.base_env.u]
        actions_to_t2 = [self.base_env.u, self.base_env.u, self.base_env.u, self.base_env.r, self.base_env.r]

        # Tenv single constraint no buffer
        self.tenv_single.reset()
        for a in actions_to_failure:
            obs, r, g, done, inf = self.tenv_single.step(a)
        assert_array_equal(g, [1, 0])
        assert not done # The fact that done is true in a death situation depends on the env and the multidiscrete one does not have that

        self.tenv_single.reset()
        for a in actions_to_t1:
            obs, r, g, done, inf = self.tenv_single.step(a)
        assert_array_equal(g, [0, 1])
        assert done

        # Tenv mult constraint no buffer
        self.tenv_multi.reset()
        for a in actions_to_failure:
            obs, r, g, done, inf = self.tenv_multi.step(a)
        assert_array_equal(g, [1, 0, 0])
        assert not done

        self.tenv_multi.reset()
        for a in actions_to_t1:
            obs, r, g, done, inf = self.tenv_multi.step(a)
        assert_array_equal(g, [0, 1, 0])
        assert done

        self.tenv_multi.reset()
        for a in actions_to_t2:
            obs, r, g, done, inf = self.tenv_multi.step(a)
        assert_array_equal(g, [0, 0, 1])
        assert done

        # Test single constraint with buffer
        self.tenv_single_buff.reset()
        for a in actions_to_failure:
            obs, r, g, done, info = self.tenv_single_buff.step(a)
        assert_array_equal(g, [1, 0])
        assert_equal(obs, (1, 1))
        assert_equal(self.base_env.state, (1, 1))
        assert not done

        self.tenv_single_buff.reset()
        for a in actions_to_t1:
            obs, r, g, done, info = self.tenv_single_buff.step(a)
        assert_array_equal(g, [0, 1])
        assert_equal(obs, (2, 1))
        assert_equal(self.base_env.state, (2, 1))
        assert not done

        # Test multiple constraints with buffer
        self.tenv_multi_buff.reset()
        for a in actions_to_failure:
            obs, r, g, done, info = self.tenv_multi_buff.step(a)
        assert_array_equal(g, [1, 0, 0])
        assert_equal(obs, (1, 1))
        assert_equal(self.base_env.state, (1, 1))
        assert not done

        self.tenv_multi_buff.reset()
        for a in actions_to_t1:
            obs, r, g, done, info = self.tenv_multi_buff.step(a)
        assert_array_equal(g, [0, 1, 0])
        assert_equal(obs, (2, 1))
        assert_equal(self.base_env.state, (2, 1))
        assert not done

        self.tenv_multi_buff.reset()
        for a in actions_to_t2:
            obs, r, g, done, info = self.tenv_multi_buff.step(a)
        assert_array_equal(g, [0, 0, 1])
        assert_equal(obs, (1, 3))
        assert_equal(self.base_env.state, (1, 3))
        assert not done

    def test_reward_shaping_with_buffer(self):
        """
        Test that potential based reward shaping is preserved when
        intervention uses state buffer. That is, if the teacher artificially
        sets the state to s'' instead of s' as dictated by the dynamics,
        the student also gets a reward phi(s'') - phi(s).
        """
        env = FrozenLakeEnvCustom(
            map_name='4x4', shaping_potential=np.arange(16).reshape(4, 4),
            shaping_coef=1, is_slippery=False)

        def constraint(**kwargs):
            return {'g': False}
        cenv = CMDP(env, constraint, [0], 1)

        def condition(info=None, **kwargs):
            return float(info['next_state_type'] in 'H')

        intervention = TeacherIntervention(cenv, [condition], [0], 2)
        actions = [DOWN, RIGHT]

        intervention.reset()
        r = np.zeros(2, dtype=float)
        for j, a in enumerate(actions):
            _, r[j], _, _, _ = intervention.step(a)
        r_target = [4-0.01, 1  - 5]
        assert_equal(r, r_target)


class TestTeacherEnv(unittest.TestCase):
    def setUp(self):
        self.n = 5

        def constraint(observation, **kwargs):
            # Constraint on the diagonal except start and goal
            if observation[0] == observation[1] and observation[0] != 0 and observation[0] != self.n - 1:
                return {'g': 1}
            else:
                return {'g': 0}

        self.env = GridEnvConstrained(self.n, constraint,
                                      multi_discrete_action=True,
                                      goal_reward=0)
        student_cls = LagrangianStudent
        student_default_kwargs = {'env': None,
                                  'br_algo': QLearning,
                                  'online_algo': ExponetiatedGradient,
                                  'br_kwargs': {'policy': TabularQPolicy,
                                                'lr_final': 0.1,
                                                'temp_fraction': 0.8},
                                  'online_kwargs': {'B': 10, 'eta': 0.5},
                                  'curriculum_transfer': identity_transfer,
                                  'lagrangian_ronuds': 10}


        def int0(observation, **kwargs):
            return float(observation in [(1, 1), (2, 2), (3, 3)])

        def int1(observation, **kwargs):
            return float(observation in [(1, 2), (1, 3), (2, 3)])

        def int2(observation, **kwargs):
            return float(observation in [(2, 1), (3, 1), (3, 2)])

        test_env = TeacherIntervention(self.env, int0, [0], 0)

        self.condition_list =[int0, int1, int2]

        args_list = zip(self.condition_list, [[0], [0], [0]], [0, 0, 0],
                        [{'avg_constraint': False}] * 3)
        interventions = [TeacherIntervention(self.env, cond, t, b, **kwargs)
                         for cond, t, b, kwargs in args_list]

        self.T = TeacherEnv(student_cls, student_default_kwargs, interventions,
                            test_env, BaseEvaluationLogger,
                            test_episode_number=3, test_episode_timeout=20)

        # Teacher env with avg cosntraints with threshold different from
        # zero to chekc evaluation works in these cases
        args_list = zip(self.condition_list, [[0.1], [0.2], [0]], [0, 1, 2],
                        [{'avg_constraint': True},
                         {'avg_constraint': False},
                         {'avg_constraint': True}])
        interventions = [TeacherIntervention(self.env, cond, t, b, **kwargs)
                         for cond, t, b, kwargs in args_list]
        self.Tconstraint = TeacherEnv(student_cls, student_default_kwargs,
                                      interventions, test_env,
                                      BaseEvaluationLogger,
                                      test_episode_number=3,
                                      test_episode_timeout=20,
                                      normalize_obs=False)

        def action_generator():
            # First episode fails, second episode to timeout and third to goal
            action_list = [self.env.u, self.env.u, self.env.r, self.env.l, self.env.d, self.env.d, self.env.r,
                           self.env.r, self.env.u, self.env.r, self.env.u, self.env.u] + \
                          [self.env.u, self.env.d] * 10 + \
                          [self.env.r, self.env.r, self.env.r, self.env.r, self.env.u, self.env.u, self.env.u,
                           self.env.u]
            for a in action_list:
                yield a
        self.action_generator = action_generator

    def test_evaluate_interventions(self):
        # List of transitions using intervention one 4 times, intervention two 2 times, no intervention 6 times,
        # intervention zero one time, intervention one 3 times
        kwargs_list = [{'observation': (1, 2), 'done': False}] * 4 + \
                      [{'observation': (3, 1), 'done': False}] * 2 + \
                      [{'observation': (0, 4), 'done': False}] * 6 + \
                      [{'observation': (1, 1), 'done': False}] * 1 + \
                      [{'observation': (1, 3), 'done': False}] * 3
        actual = self.T.intervention_evaluator.evaluate_interventions(kwargs_list)
        target = np.zeros((16, 3, 1))
        target[:4, 1, 0] = 1
        target[4:6, 2, 0] = 1
        target[12:13, 0, 0] = 1
        target[13:, 1, 0] = 1
        assert_array_equal(actual, target)

    def test_evaluate_student(self):
        self.T.set_student_env(self.T.test_env)
        # Set the sequence of actions taken by the student during evaluation

        actions = self.action_generator()
        self.T.student.predict = lambda obs, deterministic: (next(actions), None)
        rewards, lagrangian_rewards, constraint_values, successes, lengths \
            = self.T.evaluate_student()

        assert_array_almost_equal(rewards, [-0.12, -0.2, -0.07])
        assert_array_almost_equal(lagrangian_rewards, [-10/3 * 2 -0.12, -0.2, -0.07])  # Need times 2 because intervention 0 and original constraint coincide
        assert_array_almost_equal(constraint_values, [[[1], [1], [3]], [[0], [0], [0]], [[0], [0], [0]]])
        assert_array_equal(lengths, [12, 20, 8])

    def test_state_and_reward(self):
        actions = self.action_generator()
        self.T.normalize_obs = False
        self.T.reset()
        old_metric = self.T.student_success_metric
        # Predict determines trajectory in evaluate student
        self.T.student.predict = lambda obs, deterministic: (next(actions), None)
        # Act determines the trajectories during student.learn (going up we
        # do not update the multipliers and later calculations are easier)
        self.T.student.br.act = lambda obs, update_eps: self.env.u
        self.T.learning_steps = 20
        obs, r, done, info = self.T.step(1)
        target_obs = [-0.13, 1/3, 1/3, 1]
        assert_array_almost_equal(target_obs, obs)
        assert_almost_equal(r, (-10/3 * 2 -0.12 - 0.2 - 0.07)/3 - old_metric)

        actions = self.action_generator()
        self.T.normalize_obs = True
        self.T.reset()
        old_metric = self.T.student_success_metric
        # Predict determines trajectory in evaluate student
        self.T.student.predict = lambda obs, deterministic: (
            next(actions), None)
        # Act determines the trajectories during student.learn (going up we
        # do not update the multipliers and later calculations are easier)
        self.T.student.br.act = lambda obs, update_eps: self.env.u

        obs, r, done, info = self.T.step(1)
        target_obs = [1/8 * 1/3, 1/12 * 1/3, 1/12 * 1/3, 3/12 * 1/3]
        assert_array_almost_equal(target_obs, obs)
        assert_almost_equal(r, (-10 / 3 * 2 - 0.12 - 0.2 - 0.07) / 3 -
                            old_metric)

    def test_evaluation_different_constraints(self):
        """
        Test that evaluation works with avg constraints and non-zero
        thresholds."""
        self.Tconstraint.reset()
        actions = self.action_generator()
        self.Tconstraint.student.predict = lambda obs, deterministic: (
                next(actions), None)
        rewards, lagrangian_rewards, constraint_values, successes, lengths \
            = self.Tconstraint.evaluate_student()

        assert_array_almost_equal(rewards, [-0.12, -0.2, -0.07])
        assert_array_almost_equal(constraint_values,
                                  [[[1/12 - 0.1], [1 - 0.2], [0.25]],
                                   [[-0.1], [-0.2], [0]],
                                   [[-0.1], [-0.2], [0]]])
        assert_array_equal(lengths, [12, 20, 8])



class TestSetDictValues(unittest.TestCase):
    def test(self):
        base_dict = {'a': 1, 'b': 4, 'c': {'aa': 11, 'bb:': {'aaa': 111, 'bbb': 444, 'ccc': 555}, 'cc': 55}}
        new_vals_dict = {'aaa': 0, 'cc': 0, 'b': 0, 'd': 0}
        target = {'a': 1, 'b': 0, 'c': {'aa': 11, 'bb:' :{'aaa': 0, 'bbb': 444, 'ccc': 555}, 'cc': 0}}

        updated_dict, not_found_elements = set_dict_values(base_dict, new_vals_dict)
        assert target == updated_dict
        assert {'d': 0} == not_found_elements


if __name__ == '__main__':
    unittest.main()