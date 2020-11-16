import numpy as np
import warnings
from functools import partial

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds

from src.envs import CMDP, LagrangianMDP, LagrangianMDPMonitor
from src.CMDP_solvers.abstract import CMDPSolverBase

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

__all__ = ['LagrangianCMDPSolver', 'get_lagrangian_env']


class LagrangianCMDPSolver(CMDPSolverBase):
    """
    Class to solve CMDP with Lagrangian method.

    The method we use is bases on "Batch policy learning under constraints"
    by Le et al. The constrained MDP is addressed by solving a sequence of
    unconstrained MDPs. In particular, we alternate between a best response (BR)
    algorithm that solves the unconstrained problem deriving from fixing the
    value of the Lagrange multipliers and an online optimization algorithm
    that sets the multipliers based on the performance of the BR.
    """

    # TODO: Estimate the duality gap for stopping

    def __init__(self, env, br_algo, online_algo, br_kwargs=None, online_kwargs=None, _init_setup_model=True,
                 lagrangian_ronuds=10, log_training=False,
                 br_uses_vec_env=False, n_envs=1, use_sub_proc_env=True):
        """
        
        Parameters
        ----------
        env: src.envs.CMDP or None
        br_algo: stable baselines algorithm class
            Best response algorithm
        online_algo: src.online
            Online optimization algorithm class
        br_kwargs: dict
            Keyword arguments for best response
        online_kwargs: dict
            Keyword arguments for online opt algorithm
        _init_setup_model: bool
            Whether to set up the br and online upon initialization
        lagrangian_ronuds: int
            Number of times we alternate between br and online
        log_training: bool  
            Whether to log episode rewards and constraints during training
        br_uses_vec_env: bool
            Whether br algorithms needs a vectorized environment            
        n_envs: int 
            Number of environments to use (only relevant for vectorized case)
        use_sub_proc_env: bool
            Whether to use subprocesses for vectorized env (otherwise dummy 
            vec is used)
        """
        self.br_algo = br_algo
        self.online_algo = online_algo
            
        self.br_kwargs = {} if br_kwargs is None else br_kwargs
        online_kwargs = {} if online_kwargs is None else online_kwargs
        self.online_kwargs = online_kwargs.copy()
        
        # Initialize placeholders to fill when setting the environment and 
        # the model
        self.br = None
        self.online = None
        self.unconstrainedMDP = None  # The MDP resulting from Lagrangian ofCMDP
        
        self._env = None
        self.observation_space = None
        self.action_space = None
        self.env_generator = None
        self.lagrangian_rounds = lagrangian_ronuds
        self._log_training = log_training
        self.training_rewards = None
        self.training_constraints = None
        
        # Vectorized environment arguments        
        self.br_uses_vec_env = br_uses_vec_env
        self.use_sub_proc_env = use_sub_proc_env
        self.n_envs = n_envs

        self.set_env(env)

        if _init_setup_model:
            self.setup_model()

    def set_unconstrainedMDP(self):
        """
        Set up the unconstrained Lagrangian MDP.

        It can be set up either as a normal environment, a dummy vecotrized
        environment or a multiprocessing vectorized environment
        """
        assert self.online is not None, 'Need a value for Lagrange ' \
                                        'multipliers to initialize the ' \
                                        'unconstrained MDP'

        if self.br_uses_vec_env:
            # The function that generate the Lagrangian environment needs to
            # be outside the class to avoid pickling errors with
            # multiprocessing
            lagrangian_env = partial(get_lagrangian_env,
                                     cenv=None, # Passing _env here is not  necessary and slows down serialization a lot
                                     w=self.online.w,
                                     cenv_gen=self.env_generator)
            assert self.env_generator is not None, \
                'Environment generator is necessary for vectorized env'

            # With subprocesses for env
            if self.use_sub_proc_env:
                self.unconstrainedMDP = SubprocVecEnv(
                        [lagrangian_env for _ in range(self.n_envs)])

            # With dummy vec env
            else:
                self.unconstrainedMDP = DummyVecEnv(
                    [lagrangian_env for _ in range(self.n_envs)])
        else:
            lagrangian_env = partial(get_lagrangian_env,
                                     cenv=self._env,
                                     w=self.online.w,
                                     cenv_gen=self.env_generator)
            self.unconstrainedMDP = lagrangian_env()

    def _initialize_online(self):
        if self._env is not None:
            d = self._env.n_constraints + 1
            self.online_kwargs.update({'d': d})
            self.online = self.online_algo(**self.online_kwargs)
        else:
            print('Skipping online initialization since there is no env')

    def update_online(self, keep_multipliers=False):
        """
        Update online optimization algorithm.
        """
        if self.online is not None and keep_multipliers and \
                self._env.n_constraints + 1 == len(self.online.w):
            pass
        else:
            self._initialize_online()

    def setup_model(self):
        """
        Set best response.
        """
        if self.unconstrainedMDP is None:
            self.br = None
        else:
            br_kwargs = self.br_kwargs.copy()
            br_kwargs.update({'env': self.unconstrainedMDP})
            self.br = self.br_algo(**br_kwargs)


    def _setup_learn(self, seed):
        """
        check the environment, set the seed, and set the logger

        Parameters
        ----------
        seed: int
            The seed value
        """
        if self._env is None:
            raise ValueError("Error: cannot train the model without a valid environment, please set an environment with"
                             "set_env(self, env) method.")
        if seed is not None:
            set_global_seeds(seed)

    def learn(self, total_timesteps, seed=None, log=False):
        """
        Solve the CMDP alternating BR and online algorithm.

        Parameters
        ----------
        total_timesteps: int
            Total number of timesteps the algorithm is run for. Each
            Lagrangian round (i.e. alternation of br and online) is run to
            total_timesteps/self.lagrangian_rounds.
        seed: int or None
            The random seed
        log: Bool
            Print to screen some statistics about the BR training.

        Returns
        -------
        R: float
            Return when evaluating the policy learned by BR in last
            Lagrangian round
        G: np.ndarray
            Constraint when evaluating the policy learned by BR in last
            Lagrangian round
        w: np.ndarray
            Lagrange multipliers
        """

        self._setup_learn(seed)

        if total_timesteps < self.lagrangian_rounds:
            raise ValueError("There should be more time steps than Lagrangian rounds")

        # Number of timesteps per Lagrangian round
        br_time_steps = np.full(self.lagrangian_rounds, int(total_timesteps / self.lagrangian_rounds))
        br_time_steps[-1] += np.mod(total_timesteps, self.lagrangian_rounds)

        # Alternate between br and online
        for ts in br_time_steps:

            # Reset the monitor that tracks the performance of BR on the
            # unconstrained Lagrangian MDP (constraint violation is also
            # tracked)
            if self.br_uses_vec_env:
                self.unconstrainedMDP.env_method('reset_monitor')
            else:
                self.unconstrainedMDP.reset_monitor()
            self.br._init_num_timesteps()  # Reset exploration schedule

            # Train BR on unconstrained MDP
            if log:
                self.br.learn(ts, log_interval=ts)
            else:
                self.br.learn(ts, log_interval=np.inf)

            # Get training performance
            if self.br_uses_vec_env:
                # Get reward and constraints from all envs
                r_tmp = self.unconstrainedMDP.env_method(
                    'get_episode_rewards')
                g_tmp = self.unconstrainedMDP.env_method(
                    'get_episode_constraints')
                current_rewards = np.concatenate(r_tmp)
                current_constraints = np.concatenate(g_tmp)
            else:
                current_rewards = \
                    self.unconstrainedMDP.get_episode_rewards()
                current_constraints = \
                    self.unconstrainedMDP.get_episode_constraints()

            R = np.mean(current_rewards)
            G = np.mean(current_constraints, axis=0)

            # Log info about training
            if self._log_training:
                if self.training_rewards is None:
                    self.training_rewards = np.copy(current_rewards)
                else:
                    self.training_rewards = np.hstack((
                        self.training_rewards, current_rewards))
                # self.training_rewards.append(list(current_rewards))
                if self.training_constraints is None:
                    self.training_constraints = np.copy(current_constraints)
                else:
                    self.training_constraints = np.vstack((
                        self.training_constraints, current_constraints))

            # evaluate performance may be necessary for off-policy methods
            # where the deployed policy is different from the one that
            # collects data (in that case, it would make sense to adjust the
            # multipliers according to the optimized policy and not the
            # exploratory one)
            # R, G = self.evaluate_performance(int(0.2 * ts), min_episodes=5)


            # print('Evaluation r:{}\tEvaluation g {}'.format(R, G))

            # Online algorithm updates multipliers based on BR performance
            self.online.step(-np.append(G, 0))

            # Set new multipliers
            if self.br_uses_vec_env:
                self.unconstrainedMDP.set_attr('lam', self.online.w[:-1])
            else:
                self.unconstrainedMDP.lam = self.online.w[:-1]

        return R, G, self.online.w

    def predict(self, observation, state=None, mask=None, deterministic=True):
        """
        Get the best response action from an observation
        """
        if self.br is not None:
            return self.br.predict(observation, state, mask, deterministic)
        else:
            raise ValueError('Need a valid environment to setup learner and predict its action')

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if self.br is not None:
            return self.br.action_probability(observation, state, mask, actions, logp)
        else:
            raise ValueError('Need a valid environment to setup learner and predict its action probabilities')

    def evaluate_performance(self, min_steps, min_episodes):
        """
        Deploy policy learned by BR to evaluate its performance in terms of
        return and constraint violation.

        Parameters
        ----------
        min_steps: int
            Minimum number of steps that we run the environment for
        min_episodes: int
            Minimum number of episodes

        Returns
        -------
        R: float
            Average return across episodes
        G: np.ndarray
            Average constraint value across episods

        """
        if self.unconstrainedMDP is None:
            raise ValueError('Cannot reset monitor without a valid environment')

        n_episodes = 0
        n_steps = 0
        max_steps = min_steps * 5 #  Fix a timeout

        # TODO: If we move to subproc env, we should aim to use the
        #  vectorized env properly here

        if self.br_uses_vec_env:
            # This is equivalent to the non-vectorized case since we operate
            # only on one env. However, we still need to use the vectorized
            # env interface to access the individual attributes and methods.

            # Reser monitor and env
            self.unconstrainedMDP.env_method('reset_monitor')
            obs = self.unconstrainedMDP.env_method('reset', indices=0)[0]

            # Run env
            while (n_episodes < min_episodes or n_steps < min_steps) and not n_steps > max_steps:
                action, _ = self.br.predict(obs, deterministic=True)
                obs, reward, done, info = self.unconstrainedMDP.env_method(
                    'step', action, indices=0)[0]

                if done:
                    n_episodes += 1
                    obs = self.unconstrainedMDP.env_method('reset',
                                                           indices=0)[0]
                n_steps += 1

            # Compute return and contraint
            R = np.mean(self.unconstrainedMDP.env_method(
                'get_episode_rewards', indices=0)[0])
            G = np.mean(self.unconstrainedMDP.env_method(
                'get_episode_constraints', indices=0)[0], axis=0)
        else:
            # Reser monitor and env
            self.unconstrainedMDP.reset_monitor()
            obs = self.unconstrainedMDP.reset()

            # Run env
            while (n_episodes < min_episodes or n_steps < min_steps) and not n_steps > max_steps:
                action, _ = self.br.predict(obs, deterministic=True)
                obs, reward, done, info = self.unconstrainedMDP.step(action)

                if done:
                    n_episodes += 1
                    obs = self.unconstrainedMDP.reset()
                n_steps += 1

            # Compute return and contraint
            R = np.mean(self.unconstrainedMDP.get_episode_rewards())
            G = np.mean(self.unconstrainedMDP.get_episode_constraints(), axis=0)

        return R, G

    def set_env(self, env, keep_multipliers=False, reset_br=False):
        """
        Set a new environment.

        Parameters
        ----------
        env: src.envs.CMDP
        keep_multipliers: bool
        setup_model: bool
        """
        # Clean up resources if vectorized env already exists

        if isinstance(self.unconstrainedMDP, (DummyVecEnv, SubprocVecEnv)):
            self.unconstrainedMDP.close()

        # For vectorized environment we need an environment generating
        # function, otherwise we can simply set the env
        if self.br_uses_vec_env:
            if env is not None:
                assert callable(env), 'An environments generating callable is ' \
                                      'necessary for algorithms requiring a ' \
                                      'vectorized environment'

                # If necessary, this extra copy of the env can be removed.
                # Need to check all the places where _env is accessed and
                # modify them.
                super().set_env(env())
                self.env_generator = env
        else:
            super().set_env(env)
            self.env_generator = None  # Not needed in non-vectorized case

        if self.get_env() is not None:
            self.update_online(keep_multipliers)
            self.set_unconstrainedMDP()
            if reset_br or self.br is None:
                self.setup_model()
            self.br.set_env(self.unconstrainedMDP)

        self.training_rewards = None
        self.training_constraints = None

    def get_env(self):
        return super().get_env()

    def set_multipliers(self, w):
        if self.online is not None:
            if len(w) != len(self.online.w):
                raise ValueError('Multipliers must have the same length. Old ones have length {}, while new ones have '
                                 'length {}'.format(len(self.online.w), len(w)))
            else:
                self.online.w = w
        else:
            warnings.warn('There is no online algorithm to set the multipliers for')

    def get_multipliers(self):
        return self.online.w

    def get_br_params(self):
        return self.br.get_parameters()

    def set_br_params(self, params):
        self.br.load_parameters(params)

    def get_params(self):
        params = self.get_br_params()
        multipliers = self.get_multipliers()
        params.update({'multipliers': multipliers})
        return params

    def set_params(self, params):
        multipliers = params['multipliers']
        self.set_multipliers(multipliers)

        del params['multipliers']
        self.set_br_params(params)

    def get_training_performance(self):
        if not self._log_training:
            warnings.warn('Log training is set to False and no data was logged')

        return self.training_rewards, self.training_constraints

    @property
    def log_training(self):
        return self._log_training

    @log_training.setter
    def log_training(self, value):
        self._log_training = bool(value)


def get_lagrangian_env(cenv, w, cenv_gen=None):
    """
    Create Lagrangian MDP with performance monitor from CMDP and multipliers.
    """
    if cenv_gen is None:
        return LagrangianMDPMonitor(LagrangianMDP(cenv, np.copy(w[:-1])),
                                    allow_early_resets=True)
    else:
        return LagrangianMDPMonitor(LagrangianMDP(cenv_gen(), np.copy(w[:-1])),
                                    allow_early_resets=True)

