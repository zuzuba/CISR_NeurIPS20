from abc import ABC, abstractmethod

from src.envs import CMDP


class CMDPSolverBase(ABC):
    def __init__(self, env):
        self.observation_space = None
        self.action_space = None
        self._env = None
        self.set_env(env)

    def get_env(self):
        return self._env

    def set_env(self, env):
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.

        Parameters
        ----------
        env: (Gym Environment) The environment for learning a policy
        """
        if env is None and self._env is None:
            print("Loading a model without an environment, "
                  "this model cannot be trained until it has a valid environment.")
            return
        elif env is None:
            raise ValueError("Error: trying to replace the current environment with None")

        # sanity checking the environment
        if self.observation_space is not None:
            assert self.observation_space == env.observation_space, \
                "Error: the environment passed must have at least the same observation space as the model was trained on."
        else:
            self.observation_space = env.observation_space
        if self.action_space is not None:
            assert self.action_space == env.action_space, \
                "Error: the environment passed must have at least the same action space as the model was trained on."
        else:
            self.action_space = env.action_space
        assert isinstance(env, CMDP), 'Error: the base environment of a CMDP solver must be a CMDP'

        self._env = env

    @abstractmethod
    def setup_model(self):
        """
        Create all the functions necessary to train the model
        """
        pass

    @abstractmethod
    def learn(self, total_timesteps):
        """
        Train a policy to solve the CMDP by interacting with env for total_timesteps transitions
        """
        pass

    @abstractmethod
    def predict(self, observation, state=None, mask=None, deterministic=True):
        """
        Get the model's action from an observation
        """
        pass

    @abstractmethod
    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        """
        Get the model's action probability from an observation
        """
        pass
