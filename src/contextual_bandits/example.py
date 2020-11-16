import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from src.contextual_bandits.policies import CGPUCBPolicy
from src.contextual_bandits.trainer import ContextualBanditRL


class ContextualBanditEnv(gym.Env):
    """
    Gym environment implementing a contextual bandit problem.

    Parameters
    ---------
    n_states: int
        Number of states (i.e. contexts)
    n_actions: int
        Number of actions
    s_bounds: tuple
        min, max pair for box domains of the states (same for all states
        dimensions)
    a_bounds: tuple
        min, max pair for box domains of the actions (same for all states
        actions)
    """
    def __init__(self, n_states, n_actions, s_bounds=(-10, 10),
                 a_bounds=(-10, 10)):
        self.n_states = n_states
        self.n_actions = n_actions
        self.s_bounds = s_bounds
        self.a_bounds = a_bounds
        self.observation_space = gym.spaces.Box(*s_bounds, (n_states,))
        self.action_space = gym.spaces.Box(*a_bounds, (n_actions,))

        # Initialize objective and internal state
        self.objective = RFFObjective(n_states + n_actions, 200)
        self.state = None

    def _sample_state(self):
        """
        Sample state uniformly within the bounds.
        """
        tmp = np.atleast_2d(self.s_bounds * self.n_states)
        return tmp[:, 0] + (tmp[:, 1] - tmp[:, 0]) * np.random.rand(
            self.n_states)


    def reset(self):
        """
        Reset problem.
        """
        self.state = self._sample_state()
        return self.state

    def step(self, action):
        """
        Get reward for current action and sample new state.
        """
        x = np.hstack((np.atleast_2d(self.state), np.atleast_2d(action)))
        r = self.objective.y(x)
        self.state = self._sample_state()
        return self.state, r, False, {}


class RFFObjective(object):
    """
    Approximation of sample from GP with RBF kernel using Random Fourier
    Features.

    Parameters
    ----------
    input_dim: int
        Dimensionality of input space
    n_features: int
        Number of features to approximate the kernel
    sigma: float
        Standard deviation of Gaussian noise perturbing the observations
    """
    def __init__(self, input_dim, n_features, sigma=0.01):
        self.input_dim = input_dim
        self.n_features = n_features
        self.sigma = sigma

        # Sample weight that defines the linear function of the features
        self.coeff = np.random.standard_normal((input_dim, n_features))

        # Sample features
        self.bias = np.random.rand(1, n_features) * (2 * np.pi)
        self.w = np.random.randn(n_features)

    def _validate_input(self, x):
        """
        Make sure input has shape n_points x input_dim.
        """
        x = np.atleast_2d(x.copy())
        if x.shape[1] != self.input_dim:
            return x.T
        else:
            return x

    def phi(self, x):
        """
        Compute the features for the given input.
        """
        x_tmp = self._validate_input(x)
        return np.sqrt(2 / self.n_features) * \
               np.cos(np.dot(x_tmp, self.coeff) + self.bias)

    def f(self, x):
        """
        Function value for the given input.
        """
        return np.dot(self.phi(x), self.w)

    def y(self, x):
        """
        Noise perturbed function measurement for the given input.
        """
        return np.dot(self.phi(x), self.w) + self.sigma * \
               np.random.randn(1)


class ContextualRegret(object):
    def __init__(self, contexts, f, bounds):
        """
        Class for the computation of contextual regrets

        Parameters
        ----------
        contexts: np.array
            Array of contexts that we use to compute max or avg regret
        f: callable
            Objective function
        bounds: list of tuples
            list of (min, max) tuples for box domains of actions
        """
        self.contexts = contexts
        self.f = f
        self.bounds = bounds

        # Compute optimum for each context
        self.optima = np.zeros_like(self.contexts, dtype=float)
        self.optimizers = np.zeros_like(self.contexts, dtype=float)

        x0 = [(b[1] - b[0]) * .5 + b[0] for b in bounds]

        for i, c in enumerate(contexts):

            # By enumeration for one dimensional action spaces
            if len(x0) == 1:
                # Function with fixed context
                f_c = lambda a: -self.f(np.column_stack((np.full_like(a, c), a)))

                actions = np.linspace(*bounds[0], 1000)
                self.optima[i] = -(f_c(actions).min())
                self.optimizers[i] = actions[f_c(actions).argmin()]
            else:
                # Function with fixed context
                f_c = lambda a: -self.f(np.hstack((c, a)))

                result = minimize(f_c, x0, bounds=self.bounds)
                self.optima[i] = -result.fun
                self.optimizers[i] = result.x

        # Init storage variables
        self.regrets_per_context = np.zeros_like(self.optima)
        self.avg_regret = 0
        self.max_regret = 0

    def compute_regret_per_context(self, BO_policy):
        """
        Compute regret for suggestions along predefined contexts.

        Parameters
        ----------
        BO_policy: src.contextual_bandits.CGPUCBPolicy

        Returns
        -------
        avg_regret: np.ndarray
            Average regret across contexts
        max_regret: np.ndarray
            Max regret across contexts
        """
        for i, c in enumerate(self.contexts):
            # Get max of posterior mean function for given context
            a = BO_policy.step(c, explore=False)
            x = np.hstack((c, a))
            self.regrets_per_context[i] = self.optima[i] - self.f(x)
        self.avg_regret = self.regrets_per_context.mean()
        self.max_regret = self.regrets_per_context.max()
        return self.avg_regret, self.max_regret

    def plot_context_functions(self, contexts=None, fig=None, surface=True):
        """
        Plot the objective for specified contexts.

        Parameters
        ----------
        contexts: np.ndarray
            Contexts that we plot the function for. If none,
            uses self.contextxs
        fig: plt.Figure
            Figure where to plot.
        surface: bool
            If True plot also the surface plot for the whole objective.
        """
        # Initialize figure, optima and optimizers
        if fig is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
        else:
            ax = plt.gca()
        if contexts is None:
            optima = self.optima
            optimizers = self.optimizers
            contexts = self.contexts
        else:
            # If given different contexts than default ones, we do not plot
            # their optima.
            optima = [np.nan] * len(contexts)
            optimizers = [np.nan] * len(contexts)
        assert len(self.bounds) == 1, \
            'Cannot plot for action spaces that have more than one dimension'

        for c, y_opt, z_opt in zip(contexts, optimizers, optima):
            # Plot the function values for one context
            y = np.linspace(*self.bounds[0], 1000)
            x = np.full_like(y, c)
            f_c = lambda a: self.f(np.column_stack((x, a)))
            z = f_c(y)
            ax.plot(x, y, z, label=c)

            # Plot optima, if available
            ax.scatter(c, y_opt, z_opt)

            ax.set_xlabel('State')
            ax.set_ylabel('Action')
        if surface:
            # Plot surface of the whole objective
            x = np.linspace(contexts.min(), contexts.max(), 100)
            y = np.linspace(*self.bounds[0], 100)
            XX, YY = np.meshgrid(x, y)
            z = self.f(np.column_stack((XX.flatten(), YY.flatten())))
            ax.plot_surface(XX, YY, z.reshape(100, 100), alpha=0.5)

        plt.legend()
        plt.show()


def plot_2D_CBO(BOpolicy, s_bounds, a_bounds, contexts=None, fig=None):
    """
    Plot 2D Contextual BO.

    Plot the surface of posterior mean and the data. If contexts are
    specified, it also plots the line for the posterior mean of those
    contexts as well as their maximum.

    Parameters
    ----------
    BO_policy: src.contextual_bandits.CGPUCBPolicy
    s_bounds: tuple
        min, max values for states (i.e. contexts)
    a_bounds: tuple
        min, max values for actions (i.e. contexts)
    contexts: np.ndarray
        Contexts for which we plot the mean and its max.
    fig: plt.Figure
    """
    # Set figures
    if fig is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    else:
        ax = plt.gca(projection='3d')

    # Predict over domain and plot mean
    s = np.linspace(*s_bounds, 50)
    a = np.linspace(*a_bounds, 50)
    XX = np.column_stack([v.ravel() for v in np.meshgrid(s, a)])
    mu, sigma = BOpolicy.predict(XX)

    ax.plot_surface(XX[:, 0].reshape(50, 50),
                    XX[:, 1].reshape(50, 50),
                    mu.reshape(50, 50),
                    alpha=0.5)

    # Scatter data points
    X, Y = BOpolicy.get_XY()
    ax.scatter(X[:, 0], X[:, 1], Y[:, 0])

    # Plot posterior mean and its maximum for specified contexts
    if contexts is not None:
        for c in contexts:
            a = np.linspace(*a_bounds, 200)
            s = np.full_like(a, c)
            mu, sigma = BOpolicy.predict(np.column_stack((s, a)), with_noise=False)
            mu = np.squeeze(mu)
            ax.plot(s, a, mu)
            ind = mu.argmax()
            ax.scatter(c, a[ind], mu[ind])


def main():
    env = ContextualBanditEnv(1, 1, s_bounds=(-3, 3), a_bounds=(-3, 3))
    model = ContextualBanditRL(CGPUCBPolicy, env,
                               surrogate_config={'max_iters': 0},
                               BO_config={'beta': lambda t: 3})
    bounds = [env.a_bounds] * env.n_actions
    contexts = np.linspace(*env.a_bounds, 5)

    regret_manager = ContextualRegret(contexts,
                                      env.objective.f,
                                      bounds)
    regret_manager.plot_context_functions()

    n_it = 40
    avg_r = np.zeros(n_it)
    max_r = np.zeros(n_it)

    import time
    t = time.time()

    for i in range(n_it):
        model.learn(1)

        fig = plt.figure(1)
        plt.clf()
        plot_2D_CBO(model.step_model, env.s_bounds, env.a_bounds, fig=fig,
                    contexts=contexts)
        plt.draw()
        plt.pause(0.01)
        avg_r[i], max_r[i] = regret_manager.compute_regret_per_context(
            model.step_model)
        plt.figure(2)
        plt.clf()
        plt.plot(avg_r[:i])
        plt.plot(max_r[:i])
        plt.draw()

        print('Time elapsed {}'.format(time.time() - t))


if __name__ == '__main__':
    np.random.seed(0)
    main()
