import numpy as np
import cloudpickle
import os

from stable_baselines import logger
from stable_baselines.common import set_global_seeds


class ContextualBanditRL():

    def __init__(self, policy, env, surrogate_config={}, BO_config={},
                 verbose=1, _init_setup_model=True):
        self.policy = policy
        self.env = env
        self.verbose = verbose
        self.surrogate_config = surrogate_config
        self.BO_config = BO_config
        self.num_timesteps = 0
        self.params = None
        self.observation_space = None
        self.action_space = None

        if self.env is not None:
            self.observation_space = env.observation_space
            self.action_space = env.action_space

        # Init acess points to the underlying policy before it is initialized
        self.step_model = None
        self._train_step = None
        self.act = None
        self.proba_step = None
        self.params = None
        self.summary = None
        self.episode_reward = None

        if _init_setup_model:
            self.setup_model()

    def get_env(self):
        """
        returns the current environment (can be None if not defined)
        """
        return self.env

    def set_env(self, env):
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.

        Parameters
        ----------
        env: (Gym Environment) The environment for learning a policy
        """
        if env is None and self.env is None:
            if self.verbose >= 1:
                print("Loading a model without an environment, "
                      "this model cannot be trained until it has a valid environment.")
            return
        elif env is None:
            raise ValueError("Error: trying to replace the current environment with None")

        # sanity checking the environment
        assert self.observation_space == env.observation_space, \
            "Error: the environment passed must have at least the same observation space as the model was trained on."
        assert self.action_space == env.action_space, \
            "Error: the environment passed must have at least the same action space as the model was trained on."
        self.env = env

    def setup_model(self):
        """
        Create all the functions necessary to train the model
        """
        # Create policy object
        self.step_model = self.policy(self.observation_space,
                                      self.action_space,
                                      self.surrogate_config,
                                      self.BO_config)

        def _act(obs):
            """
            Take an action for given observation.
            """
            # Action maximizing the acquisition function
            return self.step_model.step(obs, explore=True)

        self.act = _act

        def _train_step(obses_t, actions, rewards, obses_tp1, dones, lr):
            n_transitions = len(np.atleast_1d(rewards))
            if n_transitions == 1:
                self.step_model.update(obses_t, actions, rewards)
            else:
                raise NotImplementedError

        self._train_step = _train_step

        self.proba_step = None

        # TODO: Hyperparameters if we optimize them?
        def params():
            if self.step_model.BO is not None:
                X, Y = self.step_model.get_XY()
                return {'X': X,
                        'Y': Y}
            else:
                return {'X': None, 'Y': None}
        self.params = params

    def _setup_learn(self, seed):
        """
        check the environment, set the seed, and set the logger
        :param seed: (int) the seed value
        """
        if self.env is None:
            raise ValueError("Error: cannot train the model without a valid environment, please set an environment with"
                             "set_env(self, env) method.")
        if seed is not None:
            set_global_seeds(seed)

    def get_parameter_list(self):
        """
        Return X and Y data.
        """
        return self.params()

    def get_parameters(self):
        """
        Return X and Y data.
        """
        return self.params()

    def pretrain(self, dataset, n_epochs=10, learning_rate=1e-4, val_interval=None):
        """
        Pretrain a model using behavior cloning: supervised learning given an expert dataset.
        """
        raise NotImplementedError

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, reset_num_timesteps=True):
        """
        Return a trained model.
        :param total_timesteps: (int) The total number of samples to train on
        :param seed: (int) The initial seed for training, if None: keep current seed
        :param callback: (function (dict, dict)) -> boolean function called at every steps with state of the algorithm.
            It takes the local and global variables. If it returns False, training is aborted.
        :param log_interval: (int) The number of timesteps before logging.
        :param reset_num_timesteps: (bool) whether or not to reset the current timestep number (used in logging)
        :return: (BaseRLModel) the trained model
        """

        self._setup_learn(seed)

        # Initialize variables
        episode_rewards = [0.0]
        episode_successes = []
        obs = self.env.reset()
        episode_length = 0

        for _ in range(total_timesteps):

            num_episodes = len(episode_rewards)

            if callback is not None:
                # Stop training if return value is False, not when it is None.
                if callback(locals(), globals()) is False:
                    break

            # Act
            action = self.act(np.array(obs))
            new_obs, reward, done, info = self.env.step(action)
            episode_rewards[-1] += reward

            # Update data set
            self._train_step(obs, action, reward, new_obs, done, lr=None)

            obs = new_obs

            # Restart if necesary
            if done:
                maybe_is_success = info.get('is_success')
                if maybe_is_success is not None:
                    episode_successes.append(float(maybe_is_success))

                obs = self.env.reset()
                episode_rewards.append(0.0)
                episode_length = 0

            # Performance in last 100 episodes
            if len(episode_rewards[-101:-1]) == 0:
                mean_100ep_reward = -np.inf
            else:
                mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 6)

            # Logging
            if self.verbose >= 1 and done and log_interval is not None and num_episodes % log_interval == 0:
                logger.record_tabular("steps", self.num_timesteps)
                logger.record_tabular("episodes", num_episodes)
                if len(episode_successes) > 0:
                    logger.logkv("success rate", np.mean(episode_successes[-100:]))
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.dump_tabular()

            self.num_timesteps += 1
            episode_length += 1

        return self

    def predict(self, observation, state=None, mask=None, deterministic=True):
        """
        Get the model's action from an observation
        """
        # Deterministic is for consistency with stable_baselines. In our
        # case deterministic means no exploration, i.e., maximize posterior
        # mean
        explore = np.logical_not(deterministic)
        return self.step_model.step(observation, explore=explore), None

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        raise NotImplementedError

    def save(self, save_path):
        """
        Save the current parameters to file
        :param save_path: (str or file-like object) the save location
        """
        data = {
            'surrogate_config': self.surrogate_config,
            'BO_config': self.BO_config,
            'policy': self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space
            }
        params = self.get_parameters()
        self._save_to_file(save_path, data=data, params=params)

    @staticmethod
    def _save_to_file(save_path, data=None, params=None):
        if isinstance(save_path, str):
            _, ext = os.path.splitext(save_path)
            if ext == "":
                save_path += ".pkl"

            with open(save_path, "wb") as file_:
                cloudpickle.dump((data, params), file_)
        else:
            # Here save_path is a file-like object, not a path
            cloudpickle.dump((data, params), save_path)

    def load_parameters(self, load_path_or_dict):
        """
        Load model parameters from a file or a dictionary
        Dictionary keys should be tensorflow variable names, which can be obtained
        with ``get_parameters`` function. If ``exact_match`` is True, dictionary
        should contain keys for all model's parameters, otherwise RunTimeError
        is raised. If False, only variables included in the dictionary will be updated.
        This does not load agent's hyper-parameters.
        .. warning::
            This function does not update trainer/optimizer variables (e.g. momentum).
            As such training after using this function may lead to less-than-optimal results.
        :param load_path_or_dict: (str or file-like or dict) Save parameter location
            or dict of parameters as variable.name -> ndarrays to be loaded.
        :param exact_match: (bool) If True, expects load dictionary to contain keys for
            all variables in the model. If False, loads parameters only for variables
            mentioned in the dictionary. Defaults to True.
        """

        params = None
        if isinstance(load_path_or_dict, dict):
            # Assume `load_path_or_dict` is dict of variable.name -> ndarrays we want to load
            params = load_path_or_dict
        else:
            # Assume a filepath or file-like.
            # Use existing deserializer to load the parameters
            _, params = ContextualBanditRL._load_from_file(load_path_or_dict)

        if self.step_model is not None:
            if params['X'] is None or params['Y'] is None:
                print('Loading model with empty data set')
            else:
                self.step_model.set_XY(params['X'], params['Y'])
        else:
            raise RuntimeError('Trying to load the parameters before policy instantiation')

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        """
        Load the model from file
        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Envrionment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param kwargs: extra arguments to change the model when loading
        """
        data, params = cls._load_from_file(load_path)

        model = cls(policy=data["policy"], env=None, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()

        model.load_parameters(params)

        return model

    @staticmethod
    def _load_from_file(load_path):
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".pkl"):
                    load_path += ".pkl"
                else:
                    raise ValueError(
                        "Error: the file {} could not be found".format(
                            load_path))

            with open(load_path, "rb") as file:
                data, params = cloudpickle.load(file)
        else:
            # Here load_path is a file-like object, not a path
            data, params = cloudpickle.load(load_path)

        return data, params