import numpy as np
from gym import Wrapper
import csv
import time
import os
import json


class CMDP(Wrapper):
    """
    Wrapper around open ai gym environments for constrained Markov decision processes. It is still a gyn.Env

    Given an existing open ai gym environment and a constraint functions (possibly vector valued), we create an open ai gym environment
    for CMDPs. It is still  gym.Env but we overwrite the step function to compute the value of the constraints. To keep
    the constraints as general as possible, we allow them to take as input the observation, the reward, the boolean done, the info dict
     and the environment itself. Moreover, we let them return the cosntriaint value and a possibly modified version of
     the reward to be compatible with unsafe environments that characterize unsafety with low rewarrd.
    """
    def __init__(self, env, constraints, constraints_values=None,
                 n_constraints=1, avg_constraint=False):
        """
        Parameters
        ----------
        env: gym.Env
            Basic gym environment
        constraints: callable
            Callable that takes in obs, action, r, done, info end env and
            returns a dict with 'g' key containing the value of the costs
        constraints_values: list of floats or None
            The constraints can be sum g_t > tau or 1/T sum g_t > tau. This
            array specifies the tau values, which are assumed to be zero by
            default
        n_constraints: int
            Number of constraints
        avg_constraint: bool
            If True the constraints are of the form 1/T sum g_t > tau,
            otherwise sum g_t > tau.
        """
        super().__init__(env)
        if not callable(constraints):
            raise ValueError('Constraints should be a (potentially vector valued) callable. If you have many separate '
                             'callables, create a single one with lambda function')
        self.constraints = constraints
        self.constraints_values = constraints_values if constraints_values is not None else [0.] * n_constraints
        self.n_constraints = n_constraints

        # Our focus is on the cumulative episode constraint. Therefore,
        # we use accumulators that allow us to compute return values of the
        # step function such that \sum return_val >= 0 is equivalent to the
        # specified constraint (for both avg constraint and not and
        # regarldess of the constraint value). Sometimes, the actual value
        # of g at the current step may be needed. This is recorded in latest_g
        self.latest_g = None
        self.g_accumulator = None
        self.old_cum_val = 0
        self.episode_steps = 1
        self.avg_constraint = avg_constraint

    def step(self, action):
        # Step of original env
        observation, reward, done, info = self.env.step(action)

        # Get constraint value and augment accumulator sum g_t
        return_dict = self.constraints(observation=observation, action=action, reward=reward, done=done, info=info, env=self.env)

        assert isinstance(return_dict, dict), "Constraint function should return a dictionary with 'g' key and " \
                                              "potentially 'r' key"
        g = return_dict['g']
        reward = return_dict['r'] if return_dict.get('r') is not None else reward

        if hasattr(g, '__iter__'):
            g = list(g)
        else:
            g = [g]
        g = np.array(g)
        self.latest_g = g

        if self.g_accumulator is None:
            self.g_accumulator = np.copy(g)
        else:
            self.g_accumulator += g

        # Compute new cumulative value of the constraint (sum g_t - tau)
        if self.avg_constraint:
            new_cum_val = self.g_accumulator / self.episode_steps - \
                          self.constraints_values
        else:
            new_cum_val = self.g_accumulator - self.constraints_values

        # Return value is the difference in cumulative values since this is
        # a step cost
        ret_val = new_cum_val - self.old_cum_val
        self.old_cum_val = new_cum_val
        self.episode_steps += 1

        return observation, reward, ret_val, done, info

    def reset(self, **kwargs):
        self.g_accumulator = None
        self.old_cum_val = 0
        self.episode_steps = 1
        return super().reset(**kwargs)


class LagrangianMDP(Wrapper):
    def __init__(self, constrained_env, lam):
        assert isinstance(constrained_env, CMDP), 'Lagrangian MDP can only be built from CMDP'
        super().__init__(constrained_env)
        self.reward_range =  (-float('inf'), float('inf'))
        self.lam = lam

    @property
    def lam(self):
        return self._lambda

    @lam.setter
    def lam(self, value):
        self._lambda = np.atleast_1d(value)

    def step(self, action):
        observation, reward, g, done, info = self.env.step(action)
        info.update({'reward': reward, 'g': g})
        reward -= np.inner(self.lam, np.atleast_1d(g))
        return observation, reward, done, info


class LagrangianMDPMonitor(Wrapper):
    EXT = "monitor.csv"
    file_handler = None

    def __init__(self, lagrangian_env, filename=None, allow_early_resets=False, reset_keywords=(), info_keywords=()):
        """
        A monitor wrapper for Lagrangian MDPs environments to track the episode reward, constraints length and other data.

        Inspired by the stable_baselines Monitor class

        Parameters
        ----------
        lagrangian_env: src.envs.CMDP.LagrangianMDP
        filename: str
            the location to save a log file, can be None for no log
        allow_early_resets: bool
            allows the reset of the environment before it is done
        reset_keywords: tuple
            extra keywords for the reset call, if extra parameters are needed at reset
        info_keywords: tuple
            extra information to log, from the information return of environment.step
        """

        assert isinstance(lagrangian_env, LagrangianMDP), 'This Monitor is valid only for LagrangianMDPs'
        super().__init__(lagrangian_env)

        self.t_start = time.time()
        if filename is None:
            self.file_handler = None
            self.logger = None
        else:
            if not filename.endswith(LagrangianMDPMonitor.EXT):
                if os.path.isdir(filename):
                    filename = os.path.join(filename, LagrangianMDPMonitor.EXT)
                else:
                    filename = filename + "." + LagrangianMDPMonitor.EXT
            self.file_handler = open(filename, "wt")
            self.file_handler.write('#%s\n' % json.dumps({"t_start": self.t_start,
                                                          'env_id': lagrangian_env.spec and lagrangian_env.spec.id}))
            self.logger = csv.DictWriter(self.file_handler,
                                         fieldnames=('r', 'g', 'l', 't') + reset_keywords + info_keywords)
            self.logger.writeheader()
            self.file_handler.flush()

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets

        # Initilize single episode variables
        self.rewards = None
        self.constraints = None
        self.needs_reset = True

        # Initialize variables to store across episodes
        self.episode_rewards = []
        self.episode_constraints = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {}

    @property
    def lam(self):
        """Expose lagrange multipliers of Lagrangian MDP"""
        return self.env.lam

    @lam.setter
    def lam(self, value):
        """Set lagrange multipliers of Lagrangian MDP and reset monitor values"""
        self.reset_monitor()
        self.env.lam = value

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        Parameters
        ----------
        kwargs: dict
            Extra keywords saved for the next episode. only if defined by reset_keywords

        Returns
        -------
        observation: int or float
            first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, "
                               "wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.constraints = []
        self.needs_reset = False

        # Check that all necessary keywords are passed
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError('Expected you to pass kwarg %s into reset' % key)
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Step the environment with given action

        Parameters
        ----------
        action: int or float

        Returns
        -------
        transition: ([int] or [float], [float], [bool], dict)
            observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")

        observation, reward, done, info = self.env.step(action)

        # The true reward is contained in info because reward containes the value with the Lagrange penalty
        self.rewards.append(info['reward'])
        self.constraints.append(info['g'])
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            eplen = len(self.rewards)
            ep_constraints = list((np.sum(np.array(self.constraints), axis=0)))
            ep_info = {"r": round(ep_rew, 6), "g":ep_constraints, "l": eplen, "t": round(time.time() - self.t_start, 6)}

            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_rewards.append(ep_rew)
            self.episode_constraints.append(ep_constraints)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.t_start)

            ep_info.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(ep_info)
                self.file_handler.flush()
            info['episode'] = ep_info

        self.total_steps += 1
        return observation, reward, done, info

    def reset_monitor(self):
        self.t_start = time.time()
        self.episode_rewards = []
        self.episode_constraints = []
        self.episode_lengths = []
        self.total_steps = 0

    def close(self):
        """
        Closes the environment
        """
        if self.file_handler is not None:
            self.file_handler.close()

    def get_total_steps(self):
        """
        Returns the total number of timesteps
        :return: (int)
        """
        return self.total_steps

    def get_episode_rewards(self):
        """
        Returns the rewards of all the episodes
        :return: ([float])
        """
        if self.episode_rewards:
            return np.atleast_1d(self.episode_rewards)
        else:
            return np.atleast_1d(sum(self.rewards))

    def get_episode_constraints(self):
        """
        Returns the constraints of all the episodes
        :return: list of list of floats
        """
        if self.episode_constraints:
            return np.atleast_2d(self.episode_constraints)
        else:
            return np.atleast_2d(list((np.sum(np.array(self.constraints), axis=0))))


    def get_episode_lengths(self):
        """
        Returns the number of timesteps of all the episodes
        :return: ([int])
        """
        return np.atleast_1d(self.episode_lengths)

    def get_episode_times(self):
        """
        Returns the runtime in seconds of all the episodes
        :return: ([float])
        """
        return np.atleast_1d(self.episode_times)
