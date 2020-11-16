import numpy as np
from gym.spaces import Discrete, MultiDiscrete
import itertools
import copy


__all__ = ['TabularQPolicy', 'Qdict2array']


class TabularPolicy():
    def __init__(self, ob_space, ac_space, n_env=1, n_steps=1, n_batch=1):
        """
        The base policy object

        Parameters
        ----------
        ob_space: gym.space
            The observation space of the environment
        ac_space: gym.space
            The action space of the environment
        n_env: int
            The number of environments to run  (not used, just for compatibility)
        n_steps: int
            The number of steps to run for each environment (not used, just for compatibility)
        n_batch: int
            The number of batches to run (n_envs * n_steps) (not used, just for compatibility)
        """
        self.n_env = n_env
        self.n_steps = n_steps
        self.n_batch = n_batch
        self.ob_space = ob_space
        self.ac_space = ac_space

        # Validate spaces
        assert isinstance(ob_space, (Discrete, MultiDiscrete)), 'Observation space must be discrete or multi discrete for ' \
                                                              'tabular policy'
        assert isinstance(ac_space, (Discrete, MultiDiscrete)), 'Observation space must be discrete or multi discrete for ' \
                                                              'tabular policy'
        self.s = ob_space.nvec if isinstance(ob_space, MultiDiscrete) else ob_space.n
        self.a = ac_space.nvec if isinstance(ob_space, MultiDiscrete) else ac_space.n

        self.s_is_multi = isinstance(self.ob_space, MultiDiscrete)
        self.a_is_multi = isinstance(self.ac_space, MultiDiscrete)

        self.n_states = np.prod (self.s)
        self.n_actions = np.prod(self.a)

        # Initialize a uniform policy
        if self.s_is_multi:
            self.pi = {k: np.full(self.n_actions, 1/self.n_actions) for k in itertools.product(
                *[list(range(i)) for i in self.s])}
        else:
            self.pi = {k: np.full(self.n_actions, 1 / self.n_actions) for k in range(self.s)}

    def proba_step(self, obs):
        """
        Returns the action probability for a single step

        Parameters
        ----------
        obs: int or tuple of ints
            Current observation of the environment
        """
        return self.pi[obs]

    def step(self, obs, deterministic=False):
        """
        Returns the policy for a single step.

        Parameters
        ----------
        obs: int or tuple of ints
            Current observation of the environment.
        deterministic: bool
             Whether or not to return deterministic actions.

        Returns
        -------

        """
        p = self.proba_step(obs)
        if deterministic:
            ind = np.argmax(p)
        else:
            ind = int(np.nonzero(np.random.multinomial(1, p))[0])

        if self.a_is_multi:
            return list(np.unravel_index(ind, self.a))
        else:
            return ind


class TabularQPolicy(TabularPolicy):
    def __init__(self, ob_space, ac_space, n_env=1, n_step=1, n_batch=1, temperature=1, initial_Q=None):
        super().__init__(ob_space, ac_space, n_env, n_step, n_batch)

        if initial_Q is None:
            if self.s_is_multi:
                self.Q = {k: np.zeros(self.n_actions) for k in itertools.product(*[list(range(i)) for i in self.s])}
            else:
                self.Q = {k: np.zeros(self.n_actions) for k in range(self.s)}
        else:
            self.Q = copy.deepcopy(initial_Q)
        self._temperature = temperature

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    def _update_single_Q(self, s, a, val):
        """
        Update Q value for a single state action pair.

        Parameters
        ----------
        s: int or tuple of ints
            State where to  update the value
        a: int or tuple of ints
            Action where to update the value
        val: float
            New Q value

        """
        assert len(np.atleast_1d(val)) == 1, 'This function can only update the Q for one state action pair at a time'

        if self.a_is_multi:
            a = np.ravel_multi_index(a, self.a)

        val = np.atleast_1d(val)
        self.Q[s][a] = val

    def update_Q(self,s, a, val):
        """
        Update Q value and greedy policy for one or more state action pairs.

        Parameters
        ----------
        s: int or tuple of ints (or array of ints or tuple of ints)
            States where to  update the value
        a: int or tuple of ints (or array of ints or tuple of ints)
            Actions where to update the value
        val: float (or arary of floats)
            New Q value
        """
        if len(np.atleast_1d(val)) == 1:
            self._update_single_Q(s, a, val)
        else:
            for s_it, a_it, val_it in zip(s, a, val):
                self._update_single_Q(s_it, a_it, val_it)

    def proba_step(self, obs):
        """
        Returns the action probability for a single step

        Parameters
        ----------
        obs: int or tuple of ints
            Current observation of the environment
        """
        # Update the policy probability with the current temperature
        Q_vals = self.Q[obs][:]
        tmp = np.exp(Q_vals / self._temperature)
        self.pi[obs][:] = tmp / np.sum(tmp)
        return self.pi[obs]

    def update_full_policy(self):
        """Update the full policy using current temperature value"""
        for obs in self.Q.keys():
            self.proba_step(obs)

    def get_Q(self, s, a=None):
        if a is None:
            return self.Q[s]
        else:
            if self.a_is_multi:
                a = np.ravel_multi_index(a, self.a)
            return self.Q[s][a]


def Qdict2array(Qdict):
    keys = list(Qdict.keys())
    keys.sort()
    Q = []
    for k in keys:
        Q.append(Qdict[k])
    return np.array(Q)
