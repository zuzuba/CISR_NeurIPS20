import numpy as np
import unittest
import GPy
import gym.spaces
from numpy.testing import *

from src.contextual_bandits.policies import dict2ker, space2domain, CGPUCBPolicy
from src.contextual_bandits.trainer import ContextualBanditRL
from src.envs import ReachCenter


class TestPolicyHelpers(unittest.TestCase):
    def test_space2domain(self):
        """
        Test converstion of gym space into GPyOpt domain.
        """
        space = gym.spaces.Box(0, 1, (3, 2), dtype=np.float32)
        domain = [{'name': 'var_{}'.format(i),
                   'type': 'continuous',
                   'domain': (0, 1),
                   'dimensionality': 1} for i in range(6)]
        assert domain == space2domain(space)

        space = gym.spaces.Box(0, 1, (3,), dtype=np.int16)
        domain = [{'name': 'var_{}'.format(i),
                   'type': 'discrete',
                   'domain': (0, 1),
                   'dimensionality': 1} for i in range(3)]
        assert domain == space2domain(space)

        space = gym.spaces.Discrete(5)
        domain = [{'name': 'var_0',
                   'type': 'discrete',
                   'domain': np.array([0, 1, 2, 3, 4]),
                   'dimensionality': 1}]
        test_domain = space2domain(space)[0]
        for k, v in domain[0].items():
            assert np.all(v == test_domain[k]) # Need to use np.all because domain is an array

        space = gym.spaces.MultiDiscrete((5, 3))
        domain = [{'name': 'var_{}'.format(i),
                   'type': 'discrete',
                   'domain': np.arange(n),
                   'dimensionality': 1} for i, n in enumerate([5, 3])]
        test_domain = space2domain(space)
        for d, test_d in zip(domain, test_domain):
            for k, v in d.items():
                assert np.all(v == test_d[k])

    def test_dict2ker(self):
        """
        Test Kernel creation from dictionary.
        """
        d = {'type': GPy.kern.RBF,
             'kwargs': {
                 'input_dim': {'v': 1},
                 'lengthscale': {'v': 0.1, 'prior': GPy.priors.LogGaussian,
                                 'prior_params': {'mu': 0.1,
                                                  'sigma': 0.2},
                                 'constraint': 'bounded',
                                 'constraint_params': {'lower': 0,
                                                      'upper': 2}}
             }}
        k = dict2ker(d)
        assert isinstance(k, GPy.kern.RBF)
        assert k.input_dim == 1
        assert k.lengthscale.values[0] == 0.1

        prior = k.lengthscale.priors.properties()[0]
        assert isinstance(prior, GPy.priors.LogGaussian)
        assert_array_equal([prior.mu, prior.sigma], [0.1, 0.2])
        constraint = k.lengthscale.constraints.properties()[0]
        assert_array_equal([constraint.lower, constraint.upper], [0, 2])


class TestUCBPolicy(unittest.TestCase):
    def setUp(self):
        self.pi = CGPUCBPolicy(gym.spaces.Box(-10, 10, (1,), dtype=np.float64),
                               gym.spaces.Box(-10, 10, (1,), dtype=np.float64),
                               new_surrogate_config={'max_iters': 0})

        # Insert a data point so that the internal BO object is created
        self.pi.update(0, 0, 1)
        self.pi.update(1, 1, -1)

        # Create a separate GP to manually test updates and suggestions
        k = self.pi.model.kernel.copy()
        lik = self.pi.model.lik.copy()
        X, Y = self.pi.get_XY()
        self.gp = GPy.core.GP(X, Y, k, lik)

    def test_predict(self):
        """
        Test prediction function.
        """
        for state in [-1, 0, 1]:
            # Create domain
            a = np.linspace(-1, 1, 100)
            s = np.full_like(a, state)
            X = np.column_stack((s, a))

            # Compare predictions
            mu, sigma = self.pi.predict(X)
            mu_target, sigma_target = self.gp.predict_noiseless(X, full_cov=False)
            sigma_target = np.sqrt(sigma_target)
            assert_allclose(mu, mu_target, atol=1e-5)
            assert_allclose(sigma, sigma_target, atol=1e-5)

    def test_get_XY(self):
        """
        Test interface to access the data.
        """
        X, Y = self.pi.get_XY()
        X_target = np.array([[0, 0], [1, 1]])
        Y_target = np.array([[1], [-1]])
        assert_allclose(X, X_target)
        assert_allclose(Y, Y_target)
        assert_allclose(self.pi.BO.X, X_target)
        assert_allclose(self.pi.BO.Y, -Y_target)

    def test_step(self):
        """
        Test policy recommendation in exploratory case and not.
        """
        # Non exploratory moves (i.e. maximize posterior mean)
        for state in [-1, 0, 1]:
            # Compute suggestion manually
            actions = np.linspace(-2, 2, 1000)
            X = np.column_stack((np.full_like(actions, state), actions))
            mu, sigma = self.gp.predict_noiseless(X)
            target_suggestion = actions[mu.argmax(axis=0)]

            # Compare
            suggested_a = self.pi.step(state, explore=False)
            assert_allclose(suggested_a, target_suggestion, atol=1e-3)

        # Exploratory moves (i.e. maximize UCB)
        for state in [-1, 0, 1]:
            # Compute suggestion manually
            actions = np.linspace(-2, 2, 2000)

            X = np.column_stack((np.full_like(actions, state), actions))
            # GPyOpt implementations uses noise to compute the acquisitions
            # function
            mu, sigma = self.gp.predict(X, include_likelihood=True)
            sigma = np.sqrt(sigma)
            target_suggestion = actions[(mu + 2 * sigma).argmax(axis=0)]

            # Compare acquisition function and suggestion
            assert_allclose(mu + 2 * sigma,
                            self.pi.BO.acquisition._compute_acq(X), atol=1e-5)
            suggested_a = self.pi.step(state, explore=True)
            assert_allclose(suggested_a, target_suggestion, atol=1e-3)

    def test_non_exploratory_suggestion(self):
        # Test that the non-exploring approximation works by setting the exploit
        # function to something you want
        def dummy_exploit_f(x):
            x = np.atleast_2d(x)
            return (x[:, 1] - 2 * x[:, 0]) ** 2
        self.pi.exploit_func = dummy_exploit_f

        assert_almost_equal(self.pi.step(1., explore=False), 2, decimal=5)
        assert_almost_equal(self.pi.step(2., explore=False), 4, decimal=5)
        assert_almost_equal(self.pi.step(-3., explore=False), -6, decimal=5)

    def test_update(self):
        """
        Test adding data in the (s, a, r) format.
        """
        # Add one data point
        self.pi.update(-1, -1, 1)
        X_target = np.array([[0, 0], [1, 1], [-1, -1]])
        Y_target = np.array([[1], [-1], [1]])
        X, Y = self.pi.get_XY()
        assert_allclose(X, X_target)
        assert_allclose(Y, Y_target)
        assert_allclose(self.pi.BO.X, X_target)
        assert_allclose(self.pi.BO.Y, -Y_target)

        # Add multiple data points
        self.pi.update(np.array([[2], [-2]]), np.array([[2], [-2]]),
                       np.array([[2], [-2]]))
        X_target = np.array([[0, 0], [1, 1], [-1, -1], [2, 2], [-2, -2]])
        Y_target = np.array([[1], [-1], [1], [2], [-2]])
        X, Y = self.pi.get_XY()
        assert_allclose(X, X_target)
        assert_allclose(Y, Y_target)
        assert_allclose(self.pi.BO.X, X_target)
        assert_allclose(self.pi.BO.Y, -Y_target)

    def test_setXY(self):
        """
        Test setting data in (X, Y) format
        :return:
        """
        newX = np.array([[2, 2]])
        newY = np.array([[-2]])
        self.pi.set_XY(newX, newY)
        assert_allclose(self.pi.BO.X, newX)
        assert_allclose(self.pi.BO.Y, -newY)

    def test_sparse(self):
        # Sparse assumes that the observations are normalized for
        # inducing points
        pi_sparse = CGPUCBPolicy(
            gym.spaces.Box(0, 1, (1,), dtype=np.float64),
            gym.spaces.Box(0, 1, (1,), dtype=np.float64),
            new_surrogate_config={'max_iters': 0,
                                  'sparse': True,
                                  'num_inducing': 50})

        pi_sparse.update(np.arange(0, 1, 0.1)[:, None],
                         np.arange(0, 1, 0.1)[:, None],
                         np.arange(0, 1, 0.1)[:, None])

        k = pi_sparse.model.kernel.copy()
        lik = pi_sparse.model.lik.copy()
        X, Y = pi_sparse.get_XY()
        gp = GPy.core.GP(X, Y, k, lik)

        for state in [0, 0.5, 1]:
            # Create domain
            a = np.linspace(0, 1, 100)
            s = np.full_like(a, state)
            X = np.column_stack((s, a))

            # # Compare predictions
            mu, sigma = pi_sparse.predict(X, with_noise=False)
            mu_target, sigma_target = gp.predict_noiseless(X, full_cov=False)
            sigma_target = np.sqrt(sigma_target)
            assert_allclose(mu, mu_target, atol=2e-4)
            assert_allclose(sigma, sigma_target, atol=4e-4)


# Too slow to be ran all the time
# class TestTrainer(unittest.TestCase):
    # """
    # Test the contextual trainer in the reach-center dummy environment.
    # """
    # def setUp(self):
    #     # Define env
    #     self.env = ReachCenter(3)
    #     self.trainer = ContextualBanditRL(CGPUCBPolicy, self.env)
    #
    # def test_learn(self):
    #     """
    #     Verify that we can recover the optimal policy.
    #     """
    #     np.random.seed(1)
    #     self.trainer.learn(30)
    #     target = [[self.env.u, self.env.r, self.env.d],
    #               [self.env.u, None, self.env.d],
    #               [self.env.u, self.env.l, self.env.d]]
    #     for i in range(3):
    #         for j in range(3):
    #             predicted_a = tuple(self.trainer.predict((i, j),
    #                                              deterministic=True)[0])
    #             target_a = target[i][j]
    #             if target_a is not None:
    #                 assert predicted_a == target_a


if __name__ == '__main__':
    unittest.main()
