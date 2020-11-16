import numpy as np
import gym

from src.teacher import TeacherEnv
from src.envs.frozen_lake.flake_constants import \
    REWARD_MAPPING as student_reward

__all__ = ['TeacherFrozenEnv', 'SmallFrozenTeacherEnv']


class TeacherFrozenEnv(TeacherEnv):
    def compute_reward(self):

        self.test_env = self.final_env
        (rewards, lagrangian_rewards, constraint_values,
         termination, lengths) = self.evaluate_student()

        # Use custom reward that uses normal reward if there is no failure and n * reward for timeout for failure
        penalty = student_reward[b'F']  # Penalty for being alive
        custom_rewards = rewards.copy()

        custom_rewards[termination == -1] = 2 * penalty * self.test_episode_timeout
        # m = termination.mean()
        m = custom_rewards.mean()

        if self.student_success_metric is None:
            self.student_success_metric = m
            return m
        else:
            r = m - self.student_success_metric
            self.student_success_metric = m
            return r


class SmallFrozenTeacherEnv(TeacherFrozenEnv):
    """
    Remove last dimension of obs since it is redundant. However, we need to
    change obs space
    """
    def __init__(self, student_cls, student_default_kwargs, interventions,
                 final_env, logger_cls, student_ranges_dict={},
                 learning_steps=4000, test_episode_number=20,
                 test_episode_timeout=200, normalize_obs=True,
                 time_steps_lim=np.inf, rounds_lim=np.inf, cost_lim=np.inf):

        super().__init__(student_cls, student_default_kwargs, interventions,
                     final_env, logger_cls, student_ranges_dict,
                     learning_steps, test_episode_number,
                     test_episode_timeout, normalize_obs,
                     time_steps_lim, rounds_lim)
        if self.normalize_obs:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,),
                                                    dtype=np.float)
        else:
            self.observation_space = gym.spaces.Box(low=np.inf, high=-np.inf,
                                                    shape=(2,),
                                                    dtype=np.float)
        self.steps_teacher_episode = 0
        self.steps_per_episode = int(time_steps_lim / learning_steps)

        # Counters for training failures
        self.student_failures = 0
        self.student_training_episodes_current_env = 0

    def reset(self):
        self.steps_teacher_episode = 0 # used for obs computation => need to
        # be reset before we compute obs
        obs = super().reset()

        # Count training failures
        self.student_failures = 0
        self.student_training_episodes_current_env = 0
        self.set_student_training_logging(True)
        return obs

    def step(self, action):
        self.steps_teacher_episode += 1
        if action != self.old_action:
            self.student_training_episodes_current_env = 0
        obs, r, done, info = super().step(action)

        # Get training failures (for original constraint)
        # TODO: This method of counting training failures strongly relies on
        #  the environment definition: the fact that it can be violated only
        #  once due to reset, otherwise we would need not only to
        #  count_nonzero. Moreover, the fact that the tolerance is zero and,
        #  therefore, checking >0 condition is sufficient. We should make
        #  this more general?
        r_student, g_student = self.get_student_training_log()
        g_student = np.asarray(g_student)
        training_failures = np.count_nonzero(
            g_student[self.student_training_episodes_current_env:, 0] > 0)
        self.student_failures += training_failures
        self.student_training_episodes_current_env = len(r_student)
        return obs, r, done, info

    def compute_obs(self, ):
        # Compute observations
        if self.normalize_obs:
            raise NotImplementedError
        else:
            obs = np.zeros(self.observation_space.shape, dtype=float)

            self.test_env = self.final_env
            (rewards, lagrangian_rewards, constraint_values,
             termination, lengths) = self.evaluate_student()
            obs[0] = np.mean(termination == 1)
            obs[1] = self.old_action if self.old_action is not None else 0

            #
            # self.test_env = self.actions[0]()
            # counters = self.evaluate_student()
            # obs[0] = np.mean(counters[0])
            #
            # self.test_env = self.actions[1]()
            # counters = self.evaluate_student()
            # obs[1] = np.mean(counters[1])

            # obs[0] = self.steps_teacher_episode / self.steps_per_episode
        return obs


class SmallFrozenTrainingObservation(SmallFrozenTeacherEnv):
    def __init__(self, student_cls, student_default_kwargs, interventions,
                 final_env, logger_cls, student_ranges_dict={},
                 learning_steps=4000, test_episode_number=20,
                 test_episode_timeout=200, normalize_obs=True,
                 time_steps_lim=np.inf, rounds_lim=np.inf, cost_lim=np.inf):

        super().__init__(student_cls, student_default_kwargs, interventions,
                     final_env, logger_cls, student_ranges_dict,
                     learning_steps, test_episode_number,
                     test_episode_timeout, normalize_obs,
                     time_steps_lim, rounds_lim)
        self.observation_space = gym.spaces.Box(low=np.inf, high=-np.inf,
                                                shape=(2,), dtype=np.float)

    def compute_obs(self):
        r, g = self.get_student_training_log()

        if r is None or g is None:
            student_training_r = 0
            student_training_interventions = 0
        else:
            start = self.student_training_episodes_current_env
            student_training_r = np.mean(r[start:])
            student_training_interventions = np.mean(
                np.sum(np.asarray(g)[start:, 1:], axis=1))

        return np.array([student_training_r, student_training_interventions])

    def compute_reward(self):
        return 0


class SmallFrozenNonStationaryBandits(SmallFrozenTeacherEnv):
    def __init__(self, student_cls, student_default_kwargs, interventions,
                 final_env, logger_cls, student_ranges_dict={},
                 learning_steps=4000, test_episode_number=20,
                 test_episode_timeout=200, normalize_obs=True,
                 time_steps_lim=np.inf, rounds_lim=np.inf, cost_lim=np.inf):

        super().__init__(student_cls, student_default_kwargs, interventions,
                     final_env, logger_cls, student_ranges_dict,
                     learning_steps, test_episode_number,
                     test_episode_timeout, normalize_obs,
                     time_steps_lim, rounds_lim)
        self.observation_space = gym.spaces.Box(low=np.inf, high=-np.inf,
                                                shape=(2,), dtype=np.float)

        # Variable to store return last time a student trained on a given
        # intervention
        self.previous_returns_interventions = np.full(len(interventions),
                                                      None, dtype=float)

    def compute_obs(self):
        """
        The teacher from 'Teacher-student curriculum learning' uses samples of
        previous return values to decide which interventions to propose.
        Therefore, we can save time if we avoid computing the state.
        """
        return np.array([0, 0])

    def compute_reward(self):
        """
        Compute teacher reward as delta in average student return for current
        intervention.
        """
        # Get training average training return in current intervention
        r, g = self.get_student_training_log()

        if r is None or g is None:
            student_training_r = 0
        else:
            start = self.student_training_episodes_current_env
            student_training_r = np.mean(r[start:])

        # If the intervention has been visited before, compute the teacher
        # reward as delta in student average return for this intervention.
        # Otherwise simpy store the base perfromance of the student for the
        # current intervention
        if self.old_action is None:  # Happens upon reset
            teacher_reward = 0
        else:
            if np.isnan(self.previous_returns_interventions[self.old_action]):
                teacher_reward = np.nan  # Flag value that is not stored by the teacher
            else:
                teacher_reward = student_training_r - \
                                 self.previous_returns_interventions[self.old_action]

            self.previous_returns_interventions[self.old_action] = student_training_r

        return teacher_reward
