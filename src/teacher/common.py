import numpy as np
import gym
from collections import deque
import warnings

from src.envs import CMDP

__all__ = ['TeacherIntervention', 'TeacherEnv', 'create_intervention',
           'student_generator']


class TeacherIntervention(CMDP):
    def __init__(self, cenv, intervention_conditions, taus=None, buffer_size=0,
                 backup_controller=None, **kwargs):
        """
        Teacher intervention which takes a CMDP and augments it with teacher.

        The original CMDP is augmented with a set of constraints that is
        (1/T) sum intervention_condition >= tau. Whether the constraint is
        cumulative or average is determined by the kwarg avg_constraint.
        Every time the teacher intervenes, the agent is either reset to the
        initial state distribution or backtracks by a number of steps that
        is equal to the buffer size.

        Parameters
        ----------
        cenv: src.envs.CMDP
            Original CMDP
        intervention_conditions: list of callables
            Instantaneour value of the left hand side of the constraint
        taus: list of floats
            RIght hand side value of the constraint
        buffer_size: int
            How many steps we should backtrack when the teacher intervenes
        backup_controller: controller (not yet implemented)
            In case we cannot set the env state, we can stir with an
            appropriate controller
        kwargs: dict
            Kwargs for CMDP
        """
        if isinstance(cenv, CMDP):
            # Get the unconstrained environment
            env = cenv.env

            # Define augmented constraints based on the teacher intervention
            if not isinstance(intervention_conditions, list):
                intervention_conditions = [intervention_conditions]
            self.interventions_conditions = intervention_conditions

            def augmented_constraints(**kwargs):
                # Extract original constraint
                return_dict = cenv.constraints(**kwargs)
                g = return_dict['g']
                g = list(g) if hasattr(g, '__iter__') else [g]

                # Add teacher
                # TODO: Decide whether the intervention condition should return a dict with g and r or just a list
                teacher_constraint = [float(cond(**kwargs)) for cond in intervention_conditions]
                return_dict['g'] = g + teacher_constraint
                return return_dict

            # Add thresholds
            taus = taus if taus is not None else [0.] * len(self.interventions_conditions)

            if not isinstance(taus, list):
                taus = list(taus)

            augmented_constraints_values  = cenv.constraints_values + taus
            n_augmented_constraints = cenv.n_constraints + len(self.interventions_conditions)

            # Init CMDP with new constraints
            super().__init__(env, augmented_constraints,
                             augmented_constraints_values,
                             n_augmented_constraints, **kwargs)

            # Init buffers and backup
            self.obs_buffer = deque([], maxlen=buffer_size)
            self.state_buffer = deque([], maxlen=buffer_size)
            self.buffer_size = buffer_size
            self.backup_controller = backup_controller
            if buffer_size > 0 and not (hasattr(self.env, 'get_state') and hasattr(self.env, 'set_state')):
                raise ValueError('The environment must implement get state and set state for the teacher to reset')

    def step(self, action):
        if self.backup_controller is not None:
            raise NotImplementedError('Not yet implemented the case with backup controller')
        else:
            # Take a normal step in the env
            obs, reward, g, done, info = super().step(action)

            # We detect whether the teacher has intervened based on the
            # value of the constraint (assuming the conditions can only
            # return values that are > 0)
            teacher_intervention = any(el > 0 for el in self.latest_g[-len(
                self.interventions_conditions):])\
                if len(self.interventions_conditions) > 0 else False

            # Reset from teacher
            if teacher_intervention:
                if self.buffer_size <= 0:
                    done = True  # Use initial state distribution
                else:
                    try:
                        phi_old = self.env.get_potential_value()
                    except AttributeError:
                        phi_old = 0
                    obs = self.obs_buffer[-1]
                    self.env.set_state(self.state_buffer[-1])
                    try:
                        phi_new = self.env.get_potential_value()
                    except AttributeError:
                        phi_new = 0
                    reward += phi_new - phi_old

            # Otherwise update the buffer
            else:
                if self.buffer_size > 0:
                    self.obs_buffer.appendleft(obs)
                    self.state_buffer.appendleft(self.env.get_state())
            info.update({'teacher_intervention': teacher_intervention})
        return obs, reward, g, done, info

    def reset(self, **kwargs):
        # Reset the buffers when the episode restarts
        self.obs_buffer = deque([], maxlen=self.buffer_size)
        self.state_buffer = deque([], maxlen=self.buffer_size)
        obs = super().reset(**kwargs)
        if self.buffer_size > 0:
            self.obs_buffer.appendleft(obs)
            self.state_buffer.appendleft(self.env.get_state())
        return obs


def create_intervention(base_cenv, interventions, taus, buf_size,
                        use_vec=False, **kwargs):
    """
    Create interventions starting from base CMDP, constraints and buffer.

    The way interventions are created are different depending on whether the
    student uses vectorized environments from stable_baselines or not. In the
    first case, we return a callable that creates the intervention to be
    efficient in the serialization process required by multiprocessing.

    Parameters
    ----------
     cenv: src.envs.CMDP
        Original CMDP
    interventions: list of callables
        Instantaneous value of the left hand side of the constraint
    taus: list of floats
        RIght hand side value of the constraint
    buf_size: int
        How many steps we should backtrack when the teacher intervenes
    use_vec: bool
        If true, the student uses vectorized envs and a callable is
        returned, otherwise a TeacherIntervention object is returned
    kwargs: dict
        TeacherIntervention kwargs

    """
    if use_vec:
        assert callable(base_cenv), 'For efficiency a callable to produce ' \
                                    'the cenv is needed'

        def intervention_fn():
            return TeacherIntervention(
                base_cenv(), interventions, taus, buf_size, **kwargs)
        return intervention_fn
    else:
        return TeacherIntervention(
                base_cenv, interventions, taus, buf_size, **kwargs)


class TeacherEnv(gym.Env):
    """
    Teacher learning environment.
    """
    def __init__(self, student_cls, student_default_kwargs, interventions,
                 final_env, logger_cls, student_ranges_dict={},
                 learning_steps=4000, test_episode_number=20,
                 test_episode_timeout=200, normalize_obs=True,
                 time_steps_lim=np.inf, rounds_lim=np.inf, cost_lim=np.inf):
        """
        Parameters
        ----------
        student_cls: src.students
            The base student class
        student_default_kwargs: dict
            The keyword arguments that are necessary to create a student
        interventions: list of Teacher Interventions
            List of CMDPs that the teacher can propose to the student
        final_env: src.envs.CMDP
            Final environment where we would like to deploy the student
        logger_cls: src.teacher.loggers
            Logger class to collect statistics about the student performance in
            the test env
        student_ranges_dict: dict
            Dictionary with ranges of certain kwargs for the student
            creation. It is used to sample students from a population as
            opposed to always the same student
        learning_steps: int
            Number of steps the student spends learning every time the
            teacher proposes a CMDP
        test_episode_number: int
            Number of episodes to run in the test env for the computation of obs
            and reward of the teacher
        test_episode_timeout: int
            Number of time steps to run in the test env before timing out
        normalize_obs: bool
            If True, we normalize the observations of the student
        time_steps_lim: int
            Max number of time steps the teacher spends with the student
        rounds_lim: int
        cost_lim:
        """
        # Set first student
        self.stuent_cls = student_cls
        self.student_default_kwargs = student_default_kwargs
        self.student_ranges_dict = student_ranges_dict
        student = student_generator(self.stuent_cls, self.student_default_kwargs, self.student_ranges_dict)
        self.student = student

        # Whether to normalize the state to be in [0, 1] ^{n_interventions + 1}
        self.normalize_obs = normalize_obs

        # Actions definitions
        self.actions = interventions

        # Initialize the evaluators that computes statistics in the test
        # environment about the value of the constraints corresponding to
        # different interventions
        self.intervention_evaluator = InterventionEvaluator(interventions)

        # Obs and action space
        self.action_space = gym.spaces.Discrete(len(interventions))
        obs_shape = (1 + len(interventions) *
                     self.intervention_evaluator.n_conditions[0],)
        if self.normalize_obs:
            self.observation_space = gym.spaces.Box(low=0, high=1,
                                                shape=obs_shape, dtype=np.float)
        else:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                    shape=obs_shape,
                                                    dtype=np.float)

        # Init Logger for the test environment
        self.logger_cls = logger_cls
        self.evaluation_logger = logger_cls(
            len(interventions), self.intervention_evaluator.n_conditions[0])

        # Set stopping conditions values and counters
        self.time_steps_lim = time_steps_lim
        self.rounds_lim = rounds_lim
        self.cost_lim = cost_lim
        self.time_steps_with_student = 0
        self.rounds_with_students = 0
        self.cost_of_interventions = 0

        self.learning_steps = learning_steps

        # Testing
        self.test_env = None
        self.final_env = final_env
        self.test_episode_number = test_episode_number
        self.test_episode_timeout = test_episode_timeout

        # Initialize student reward
        self.student_success_metric = None

        # Initialize previous action
        self.old_action = None


    def _check_student(self):
        if self.student is None:
            raise ValueError('Need a valid student, current is None')

    def set_student_env(self, env):
        """Set the environment for the student."""
        self._check_student()
        self.student.set_env(env)

    def get_student_params(self):
        """Get the parameters of the student"""
        self._check_student()
        return self.student.get_params()

    def step(self, action):
        self._check_student()
        # Set the appropriate intervention
        intervention = self.actions[action]
        if action != self.old_action:
            self.student.set_env(intervention, same_env=action==self.old_action,
                                 reset_br=True)
        self.old_action = action

        # Train the student in the corresponding environment
        self.student.learn(self.learning_steps)

        # Increase the counters for teacher budget
        self.time_steps_with_student += self.learning_steps

        # Compute observation in test env
        obs = self.compute_obs()

        # Compute r in final env
        r = self.compute_reward()
        done = self.time_steps_with_student > self.time_steps_lim
        info = self.evaluation_logger.get_counters()

        return obs, r, done, info

    def set_student(self, student):
        self.student = student
        self.student.set_env(self.actions[0])  # If the environment is not set the
        # student cannot predict actios and cant be evaluated, which is necessary for reset

    def evaluate_student(self, deterministic_policy=False):
        """
        Assess student performance in test environment.
        Parameters
        ----------
        deterministic_policy: bool
            If True, it uses a deterministic policy during the evaluation.
            Whether to use it or not depends whether the student is an
            on-policy algorithm that uses randomized policies for
            exploration (then you should evaluate the randomized policy) or
            an off-line method that depolys a deterministic policy in the end.
        """
        self._check_student()
        self.evaluation_logger.reset_counters()

        obs = self.test_env.reset()

        for ep in range(self.test_episode_number):
            for i in range(self.test_episode_timeout):

                action, _ = self.student.predict(obs, deterministic=deterministic_policy)
                obs, r, g, done, info = self.test_env.step(action)

                done |= i == self.test_episode_timeout - 1  # Make sure the the timeout is recorded as an episode end

                # Kwargs to compute  intevention conditions
                kwargs = {'observation': obs, 'action': action, 'reward': r, 'done': done, 'info': info,
                          'env': self.test_env}

                # Log transition
                self.evaluation_logger.add_transition(transition_dict=kwargs,
                                                      lagrange_multipliers=self.student.get_multipliers()[:-1],
                                                      g=g,
                                                      constraint_values=self.intervention_evaluator.evaluate_interventions(kwargs))

                if done:
                    obs = self.test_env.reset()
                    break

        counters = self.evaluation_logger.get_counters()

        return counters['rewards'], counters['lagrangian_rewards'], \
               counters['constraint_values'], counters['terminations'], \
               counters['lengths']


    def compute_obs(self):
        """
        Teacher observation.

        The teacher observation is an array that includes the avg return
        over test episodes and avg constraint values for the different
        interventions over test episodes.
        """
        # TODO: Add episode lenghts to the observations now that I am
        #  normalizing? May be useful depending on the reward. If the reward
        #  does not penalize for being alive, there is no difference between
        #  an episode where I violate the constraint 1% of the time and die
        #  immediately and one where I violate it 1% of the time and finish
        #  due to timeout.

        self.test_env = self.final_env
        (rewards, lagrangian_rewards, constraint_values,
        termination, lengths) = self.evaluate_student()

        # Normalize rewards and constrtaint visitation frequency to [0, 1]
        if self.normalize_obs:
            rmin, rmax = self.test_env.env.reward_range
            norm_rewards = rewards.copy()
            if rmax != rmin:
                norm_rewards = (norm_rewards / lengths - rmin) / (rmax - rmin)
            else:
                norm_rewards = (norm_rewards / lengths - rmin)

            _, taus, _, avg_constraint = self.intervention_evaluator.get_interventions_info()
            avg_constraint = np.asarray(avg_constraint)
            norm_constraint_values = constraint_values.copy()
            norm_constraint_values += np.asarray(taus)[None, :, :]
            if np.any(~avg_constraint):
                norm_constraint_values[:, ~avg_constraint, :] /= \
                    lengths[:, None, None]
        else:
            norm_rewards = rewards
            norm_constraint_values = constraint_values

        return np.append(norm_rewards.mean(),
                         norm_constraint_values.mean(axis=0).ravel())

    def compute_reward(self):
        """
        Teacher reward.

        The teacher reward is the delta in the student success metric,
        which is chosen to be the Lagrangian rewards.
        """
        self.test_env = self.final_env
        (rewards, lagrangian_rewards, constraint_values,
         termination, lengths) = self.evaluate_student()

        m = lagrangian_rewards.mean()

        if self.student_success_metric is None:
            self.student_success_metric = m
            return m
        else:
            r = m - self.student_success_metric
            self.student_success_metric = m
            return r

    def reset(self):
        # Reset budget with students
        self.time_steps_with_student = 0
        self.rounds_with_students = 0
        self.cost_of_interventions = 0

        # Sample new student
        student = student_generator(self.stuent_cls, self.student_default_kwargs, self.student_ranges_dict)
        self.set_student(student)

        self.old_action = None

        # Get initial student state
        obs = self.compute_obs()
        self.test_env = self.final_env
        self.student_success_metric = None  # Need to reset to avoid using value from previous iteration
        self.compute_reward()
        return obs

    def get_evaluation_logs(self):
        """Get the stats of the student evaluation in the test environemnt."""
        return self.evaluation_logger.get_counters()

    def get_student_training_log(self):
        """Get the student training log in terms of cumulative reward and constraint per training episode"""
        return self.student.get_training_performance()

    def set_student_training_logging(self, must_log):
        """Set the boolean to decide whether the student should log or not"""
        self.student.log_training = must_log


class InterventionEvaluator(object):
    """
    Class to evaluate the values of the constraints corresponding to
    different interventions for given trajectories. These values are useful
    to compute the teacher observation and reward based on the trajectories
    of the student in the test environment.
    """
    def __init__(self, interventions):
        """
        Parameters
        ----------
        interventions: list of TeacherIntervention.
        """
        if not hasattr(interventions, '__iter__'):
            interventions = [interventions]
        else:
            interventions = list(interventions)
        if callable(interventions[0]):
            interventions = [i() for i in interventions]

        # Extract all the info from the TeacherIntervention class
        self.conditions = [i.interventions_conditions for i in interventions]
        self.n_conditions = [len(c) for c in self.conditions]
        self.taus = [i.constraints_values[-n:] for i, n in
                     zip(interventions, self.n_conditions)]
        self.buffer_sizes = [i.buffer_size for i in interventions]

        # Initialize accumulators to compute constraint values
        self.g_accumulator = None
        self.old_cum_val = None
        self.episode_steps = None
        self.avg_constraint = [i.avg_constraint for i in interventions]
        self.reset_accumulators()

    def reset_accumulators(self):
        self.g_accumulator = [None for _ in range(len(self.conditions))]
        self.old_cum_val = [np.zeros(n, dtype=float) for n in self.n_conditions]
        self.episode_steps = 1

    def _evaluate_single_transition(self, **kwargs):
        output = np.zeros((len(self.conditions), self.n_conditions[0]),
                           dtype=float)

        # Loop through interventions
        for i in range(len(self.conditions)):
            # Get instantaneous g value
            g = np.array([float(cond(**kwargs)) for cond in self.conditions[i]])
            if self.g_accumulator[i] is None:
                self.g_accumulator[i] = np.copy(g)
            else:
                self.g_accumulator[i] += g

            # Compute new cumulative value of the constraint (sum g_t - tau
            # or 1/T sum g_t - tau)
            if self.avg_constraint[i]:
                new_cum_val = self.g_accumulator[i] / self.episode_steps - \
                              self.taus[i]
            else:
                new_cum_val = self.g_accumulator[i] - self.taus[i]

            # Return value is the difference in cumulative values since this is
            # a step cost
            output[i, :] = new_cum_val - self.old_cum_val[i]
            self.old_cum_val[i] = new_cum_val

        self.episode_steps += 1
        if kwargs['done']:
            self.reset_accumulators()
        return output

    def evaluate_interventions(self, kwargs_list):
        if not isinstance(kwargs_list, list):
            kwargs_list = [kwargs_list]

        output = np.zeros((len(kwargs_list), len(self.conditions),
                           self.n_conditions[0]), dtype=float)
        for i, kwargs in enumerate(kwargs_list):
            output[i] = self._evaluate_single_transition(**kwargs)
        return output

    def get_interventions_info(self):
        """Return the constraint callable, the threshould level, the buffer
        size and whether the constraints are avg or not for each
        intervention."""
        return self.conditions, self.taus, self.buffer_sizes,\
               self.avg_constraint


def student_generator(student_cls, student_default_kwargs, ranges_dict):
    current_kwargs = student_default_kwargs.copy()

    # Create the dict to update student kwargs
    new_vals_dict = {k: v[0] + np.random.rand(1) * v[1] - v[0] for k, v in ranges_dict.items()}
    current_kwargs, not_found_dict = set_dict_values(current_kwargs, new_vals_dict)

    if len(not_found_dict) > 0:
        warnings.warn('The following values cound not be set because they were not present in the default '
                      'arguments{}'.format(list(not_found_dict.keys())))

    return student_cls(**current_kwargs)


def set_dict_values(base_dict, new_vals_dict):
    for k, v in base_dict.items():
        if k in new_vals_dict.keys():
            base_dict[k] = new_vals_dict[k]
            del new_vals_dict[k]
        if isinstance(v, dict):
            set_dict_values(v, new_vals_dict)
    return base_dict, new_vals_dict