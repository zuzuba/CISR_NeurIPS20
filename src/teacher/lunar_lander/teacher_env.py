import numpy as np
import gym
from functools import partial
import time
import os

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy

from src.teacher import TeacherEnv, student_generator
from src.envs.lunar_lander.interventions import \
    LunarOrthogonalInterventionMonitored, LanderOrthogonalIntervention
from src.online_learning import ExponetiatedGradient
from src.students import identity_transfer, LagrangianStudent
import src.envs.lunar_lander.utils as utils
from src.teacher.frozen_single_switch_utils import SingleSwitchPolicy


class LunarLanderTeacherEnv(TeacherEnv):
    def __init__(self, student_cls, student_default_kwargs, interventions,
                 student_ranges_dict={}, learning_steps=4000,
                 time_steps_lim=np.inf):
        # Set first student
        self.stuent_cls = student_cls
        self.student_default_kwargs = student_default_kwargs
        self.student_ranges_dict = student_ranges_dict
        student = student_generator(self.stuent_cls,
                                    self.student_default_kwargs,
                                    self.student_ranges_dict)
        self.student = student

        # Compute teacher obs from constraint and reward
        self.set_student_training_logging(True)

        # Initialize counters
        self.tot_student_training_episodes = 0 # Just for monitoring
        self.student_training_episodes_current_env = 0 # To compute obs
        self.student_failures = 0 # For performance comparison

        # Actions definitions
        self.actions = interventions

        # Obs and action space
        self.action_space = gym.spaces.Discrete(len(interventions))
        self.observation_space = gym.spaces.Box(low=0, high=np.inf,
                                                shape=(2,),
                                                dtype=np.float)

        # Set stopping conditions values and counters
        self.time_steps_lim = time_steps_lim
        self.time_steps_with_student = 0

        self.learning_steps = learning_steps

        # Initialize previous action
        self.old_action = None

    def step(self, action):
        self._check_student()
        # Set the appropriate intervention
        if action != self.old_action:
            intervention = self.actions[action]
            self.student.set_env(intervention,
                                 same_env=action==self.old_action,
                                 reset_br=True)
            self.student_training_episodes_current_env = 0
        self.old_action = action

        # Train the student in the corresponding environment
        self.student.learn(self.learning_steps)

        # Increase the counters for teacher budget
        self.time_steps_with_student += self.learning_steps

        # Compute observation in test env
        obs = self.compute_obs()

        # Compute r
        r = self.compute_reward()
        done = self.time_steps_with_student > self.time_steps_lim
        info = {}

        r_student, g_student = self.get_student_training_log()
        training_failures = np.sum(
            np.asarray(g_student)[self.student_training_episodes_current_env:, 0])
        self.student_failures += training_failures

        self.student_training_episodes_current_env = len(r_student) # This
        # needs to be updated at the end because compute_obs uses it

        return obs, r, done, info

    def evaluate_student(self, deterministic_policy=False):
        raise NotImplementedError('We have not implemented the logger for'
                                  'the lander teacher env and, therefore, '
                                  'we cannot evaluate the student')

    def compute_obs(self):
        # We could also access the Monitored env but there is no predefined
        # interface for that. We should go to
        # self.student.unconstrainedMDP.env_method (since it is a
        # SubProcVecEnv) and this would call the methods of the Lagrangian
        # Monitor, whose env is a Lagrangian MDP, whose env is a
        # MonitoredIntervention.
        r, g = self.get_student_training_log()

        if r is None or g is None:
            student_training_r = 0
            student_training_interventions = 0
        else:
            start = self.student_training_episodes_current_env
            student_training_r = np.mean(r[start:])
            student_training_interventions = np.mean(
                np.sum(np.asarray(g)[start:, 1:], axis=1))

        return [student_training_r, student_training_interventions]

    def compute_reward(self):
        return 0

    def reset(self):
        # Reset budget with students
        self.time_steps_with_student = 0

        # Sample new student
        student = student_generator(self.stuent_cls, self.student_default_kwargs, self.student_ranges_dict)
        self.set_student(student)
        self.set_student_training_logging(True)

        # Initialize counters
        self.tot_student_training_episodes = 0  # Just for monitoring
        self.student_training_episodes_current_env = 0  # To compute obs
        self.student_failures = 0  # For performance comparison

        self.old_action = None

        # Get initial student state
        obs = self.compute_obs()
        return obs

    def get_evaluation_logs(self):
        raise NotImplementedError('We have not implemented the logger for'
                                  'the lander teacher env')

    def get_student_training_log(self):
        """Get the student training log in terms of cumulative reward and constraint per training episode"""
        return self.student.get_training_performance()

    def set_student_training_logging(self, must_log):
        """Set the boolean to decide whether the student should log or not"""
        self.student.log_training = must_log


class SingleSwitchWrapper(LunarLanderTeacherEnv):
    """
    Add computation of the final performance in the original lander env.
    Make sure we record the number of failures during training time
    """

    def __init__(self, student_cls, student_default_kwargs, interventions,
                 student_ranges_dict={}, learning_steps=4000,
                 time_steps_lim=np.inf, final_env_f=None):
        super().__init__(student_cls, student_default_kwargs, interventions,
                         student_ranges_dict, learning_steps, time_steps_lim)
        self.available_actions = list(range(len(interventions)))
        if final_env_f is None:
            self.final_env_f = partial(LunarOrthogonalInterventionMonitored,
                                       cond_c=None, mod_c1=None, timeout=2000)
        else:
            self.final_env_f = final_env_f

    def train_student(self, thresholds):
        done = False
        obs = self.reset()
        action_sequence = []
        while not done:
            if obs[0] >= thresholds[0] and obs[1] <= thresholds[1] and \
                    len(self.available_actions) > 1:
                self.available_actions.pop(0)
            action = self.available_actions[0]
            action_sequence.append(action)
            obs, r, done, info = super().step(action)
        return action_sequence

    def student_final_performance(self, timesteps=int(1e5)):
        # Here we run the original env for 200.000 steps and record the
        # performance
        env = self.final_env_f()
        obs = env.reset()
        for i in range(timesteps):
            action, _ = self.student.predict(obs, deterministic=True)
            obs, r, g, done, info = env.step(action)
            if done:
                obs = env.reset()

        r =     np.array(env.get_episode_rewards())
        succ =  np.array(env.get_episode_successes())
        crash = np.array(env.get_episode_crashes())
        oom =   np.array(env.get_episode_oom())
        to =    np.array(env.get_episode_timeouts())
        tog =   np.array(env.get_episode_timeouts_on_ground())

        return r, succ, crash, oom, to, tog

    def evaluate_policy(self, thresholds, timesteps=int(1e5),
                        return_all_metric=False):
        action_sequence = self.train_student(thresholds)
        r, succ, crash, oom, to, tog = self.student_final_performance(timesteps)
        if return_all_metric:
            return r, succ, crash, oom, to, tog, action_sequence, self.student_failures
        else:
            return r


def create_single_switch_env(original=False, sensor_noise=None):
    br_kwargs = dict(policy=MlpPolicy, learning_rate=5e-3, n_steps=500,
                     ent_coef=1e-3, noptepochs=32,
                     policy_kwargs={'net_arch': [20, 20]})
    online_kwargs = dict(B=120, eta=1.0)
    student_default_kwargs = {'env': None,
                              'br_algo': PPO2,
                              'online_algo': ExponetiatedGradient,
                              'br_kwargs': br_kwargs,
                              'online_kwargs': online_kwargs,
                              'lagrangian_ronuds': 2,
                              'curriculum_transfer': identity_transfer,
                              'br_uses_vec_env': True,
                              'use_sub_proc_env': True,
                              'n_envs': 6}
    base_env_kwargs = {'sensor_noise': sensor_noise}
    if original:
        interventions = [partial(LanderOrthogonalIntervention,
                                 cond_c=None, mod_c1=None, timeout=2000,
                                 base_env_kwargs=base_env_kwargs)]
    else:
        narrow_interv = partial(LanderOrthogonalIntervention,
                                cond_c=20, mod_c1=100,
                                base_env_kwargs=base_env_kwargs)
        wide_interv = partial(LanderOrthogonalIntervention,
                              cond_c=0.5, mod_c1=1,
                              base_env_kwargs=base_env_kwargs)
        interventions = [narrow_interv, wide_interv]

    final_env_f = partial(LunarOrthogonalInterventionMonitored,
                          cond_c=None, mod_c1=None, timeout=2000,
                          base_env_kwargs=base_env_kwargs)

    teacher_env = SingleSwitchWrapper(LagrangianStudent,
                                      student_default_kwargs,
                                      interventions,
                                      student_ranges_dict={},
                                      learning_steps=int(1e5),
                                      time_steps_lim=int(1.5e6),
                                      final_env_f=final_env_f)
    return teacher_env


def create_teacher_env(original=False, sensor_noise=None, n_layers=2,
                       B=120, time_steps_lim=int(1.5e6)):

    br_kwargs = dict(policy=MlpPolicy, learning_rate=5e-3, n_steps=500,
                     ent_coef=1e-3, noptepochs=32,
                     policy_kwargs={'net_arch': [20] * n_layers})
    online_kwargs = dict(B=B, eta=1.0)
    student_default_kwargs = {'env': None,
                              'br_algo': PPO2,
                              'online_algo': ExponetiatedGradient,
                              'br_kwargs': br_kwargs,
                              'online_kwargs': online_kwargs,
                              'lagrangian_ronuds': 2,
                              'curriculum_transfer': identity_transfer,
                              'br_uses_vec_env': True,
                              'use_sub_proc_env': True,
                              'n_envs': 6}
    base_env_kwargs = {'sensor_noise': sensor_noise}
    if original:
        interventions = [partial(LanderOrthogonalIntervention,
                                 cond_c=None, mod_c1=None, timeout=2000,
                                 base_env_kwargs=base_env_kwargs)]
    else:
        narrow_interv = partial(LanderOrthogonalIntervention,
                                cond_c=20, mod_c1=100,
                                base_env_kwargs=base_env_kwargs)
        wide_interv = partial(LanderOrthogonalIntervention,
                              cond_c=0.5, mod_c1=1,
                              base_env_kwargs=base_env_kwargs)
        interventions = [narrow_interv, wide_interv]

    student_final_env_f = partial(LunarOrthogonalInterventionMonitored,
                          cond_c=None, mod_c1=None, timeout=2000,
                          base_env_kwargs=base_env_kwargs)
    teacher_env = LunarLanderTeacherEnv(LagrangianStudent,
                                        student_default_kwargs,
                                        interventions,
                                        student_ranges_dict={},
                                        learning_steps=int(1e5),
                                        time_steps_lim=time_steps_lim)
    return teacher_env, student_final_env_f


def evaluate_single_switch_policy(policy, teacher_env, student_final_env,
                                  timesteps=int(1e5)):
    # Train the student
    done = False
    obs = teacher_env.reset()
    action_sequence = []
    while not done:
        action, _ = policy.predict(obs)
        action_sequence.append(action)
        obs, r, done, info = teacher_env.step(action)

    # Deploy in final env
    student = teacher_env.student
    obs = student_final_env.reset()
    for i in range(timesteps):
        action, _ = student.predict(obs, deterministic=True)
        obs, r, g, done, info = student_final_env.step(action)
        if done:
            obs = student_final_env.reset()

    r = np.array(student_final_env.get_episode_rewards())
    succ = np.array(student_final_env.get_episode_successes())
    crash = np.array(student_final_env.get_episode_crashes())
    oom = np.array(student_final_env.get_episode_oom())
    to = np.array(student_final_env.get_episode_timeouts())
    tog = np.array(student_final_env.get_episode_timeouts_on_ground())

    return r, succ, crash, oom, to, tog, action_sequence, \
           teacher_env.student_failures

