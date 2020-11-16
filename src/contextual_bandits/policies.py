import GPy
import GPyOpt
import numpy as np
import copy
from gym.spaces import *

from GPy.kern import Matern52, RBF
from GPyOpt.models.gpmodel import GPModel, GPModel_MCMC
from GPyOpt.optimization.acquisition_optimizer import ContextManager, AcquisitionOptimizer
from GPyOpt.methods.bayesian_optimization import BayesianOptimization

__all__ = ['CGPUCBPolicy', 'dict2ker', 'space2domain',
           'DEFAULT_BO_CONFIG', 'DEFAULT_SURROGATE_CONFIG']


DEFAULT_SURROGATE_CONFIG = {
    # Kernel over observations
    'ob_ker': {'type': RBF,
               'kwargs': {
                     'lengthscale': {'v': 1.,
                                     'prior': GPy.priors.Gamma,
                                     'prior_params': {'E': 1.,
                                                      'V': 0.5}},
                     'variance': {'v': 1.,
                                     'prior': GPy.priors.Gamma,
                                     'prior_params': {'E': 1.,
                                                      'V': 0.5}}
                 }},
    # Kernel over acitons
    'ac_ker': {'type': RBF,
                 'kwargs': {
                     'lengthscale': {'v': 1.,
                                     'prior': GPy.priors.Gamma,
                                     'prior_params': {'E': 1.,
                                                      'V': 0.5}},
                     'variance': {'v': 1.,
                                     'prior': GPy.priors.Gamma,
                                     'prior_params': {'E': 1.,
                                                      'V': 0.5}}
                 }},
    # Likelihood
    'lik': {'type': GPy.likelihoods.Gaussian,
                         'kwargs': {
                             'variance': {'v': 0.0001,
                                          'prior': GPy.priors.Gamma,
                                          'prior_params': {'E': 0.01,
                                                           'V': 0.01}}
                         }},
    # How to compose the observation kernel and the action one
    'composition': 'prod',

    # Wheter to use a sparse GP model (does not work with MCMC)
    'sparse': False,
    'num_inducing': 10,

    # Whether to use MCMC to marginalize the hyperparameters
    'MCMC': False,

    # Number of restarts or iterations per restart for MLE estimate of
    # hyperparamers if MCMC is False
    'optimize_restarts': 5,
    'max_iters': 0,

    # Parameters for Markov chain and Monte Carlo estimate if MCMC True
    'n_samples': 10,
    'subsample_interval': 10,
    'n_burnin': 100,
    'step_size': 0.1
}

DEFAULT_BO_CONFIG = {
    'beta': lambda t: 2,
    'maximize': True,
    'model_update_interval': 1,
    'normalize_Y': False
}


class CGPUCBPolicy(object):
    """
    Class for Contextual GPUCB policy.

    This class implements a myopic policy using contextual GPUCB. In this
    case, the state of the system is used as a context and the actions as
    arms in a multi-armed bandit problem. It can be effective when long term
    planning is not relevant to solve the task.

    ob_space: gym.spaces
        Observation space of the MDP
    ac_space: gym.spaces
        Action space of the MDP
    new_surrogate_config: dict
        Dictionary to update the default config of the surrogate model. See
        src.contextual_bandits.policies for the default.
    new_BO_config: dict
        Dictionary to update the default config of the BO problem. See
        src.contextual_bandits.policies for the default.
    """
    def __init__(self, ob_space, ac_space, new_surrogate_config={},
                 new_BO_config={}, n_env=1, n_steps=1, n_batch=1):
        self.ob_space = ob_space
        self.ac_space = ac_space

        # Define BO domain from gym spaces
        ob_dom = space2domain(ob_space, 'obs')
        ac_dom = space2domain(ac_space, 'act')
        self.domain = ob_dom + ac_dom

        # Update the configs
        gp = copy.deepcopy(DEFAULT_SURROGATE_CONFIG)
        gp.update(new_surrogate_config)
        BO_config = copy.deepcopy(DEFAULT_BO_CONFIG)
        BO_config.update(new_BO_config)

        # Initialize kernels and GP model
        gp['ob_ker']['kwargs']['input_dim'] = {'v': len(ob_dom)}
        gp['ob_ker']['kwargs']['active_dims'] = {'v': np.arange(len(ob_dom))}
        gp['ac_ker']['kwargs']['input_dim'] = {'v': len(ac_dom)}
        gp['ac_ker']['kwargs']['active_dims'] = {'v': np.arange(
            len(ob_dom), len(ob_dom) + len(ac_dom))}

        ob_ker = dict2ker(gp['ob_ker'])
        ac_ker = dict2ker(gp['ac_ker'])
        lik = dict2ker(gp['lik'])

        if gp['composition'] == 'prod':
            ker = ob_ker * ac_ker
        elif gp['composition'] == 'sum':
            ker = ob_ker + ac_ker
        else:
            raise ValueError('Only product and sum composition implemented')

        if gp['MCMC']:
            self.model = CustomGPModel_MCMC(
                kernel=ker, lik=lik, verbose=False, n_samples=gp['n_samples'],
                subsample_interval=gp['subsample_interval'],
                n_burnin=gp['n_burnin'], step_size=gp['step_size'])
            self.acquisition_type = 'LCB_MCMC'
            self.sparse = False
        else:
            self.model = CustomGPModel(
                kernel=ker, lik=lik, sparse=gp['sparse'], num_inducing=gp[
                'num_inducing'], optimize_restarts=gp['optimize_restarts'],
                max_iters=gp['max_iters'], verbose=False)
            self.sparse = gp['sparse']
            self.acquisition_type = 'LCB'

        # Bayesian optimization
        self.BO = None
        self.beta = BO_config['beta']
        self.maximize = BO_config['maximize']
        self.model_update_interval = BO_config['model_update_interval']
        self.normalize_Y = BO_config['normalize_Y']

        # Optimization problem when we are performing pure exploitation
        self.exploit_func = None
        self.exploit_optimizer = None

    def step(self, obs, explore=True):
        """
        Returns the policy for a single step.

        Parameters
        ----------
        obs:
            Observation from env.
        explore: bool
             If True, it suggests also exploratory actions.
        """
        # We need at least one data point before creating the BO model.
        # Until then, we sample from the action space
        if self.BO is None:
            return self.ac_space.sample()
        else:
            # Add new data to gp and optimize hyperparameters if max_iters > 0
            self.BO._update_model(self.BO.normalization_type)

            # Create context dict
            if not hasattr(obs, '__iter__'):
                obs = np.array([obs])
            obs = np.asarray(obs)
            obs_copy = obs.copy()
            context = {'obsvar_{}'.format(i): v for i, v in
                       enumerate(obs_copy.flatten())}

            # Maximizing the UCB criterion trades-off exploration/exploitation
            if explore:
                self.BO.num_acquisitions += 1
                self.BO.acquisition.exploration_weight = self.beta(self.BO.num_acquisitions)
                self.BO.context = context
                x = self.BO._compute_next_evaluations()

            # In case we don't explore, we maximize the posterior mean
            else:
                # Optimize the exploit function (posterior mean)
                self.exploit_optimizer.context_manager = ContextManager(
                    space=self.BO.space, context=context)
                x, fx = self.exploit_optimizer.optimize(f=self.exploit_func)

            # Remove context (i.e. state) and cast to correct type
            x = np.squeeze(x)
            x_no_context = x[len(obs):]
            if isinstance(self.ac_space, Discrete):
                x_no_context = int(x_no_context[0])
            elif isinstance(self.ac_space, MultiDiscrete):
                x_no_context.astype(np.int64)

            return x_no_context

    def update(self, obs, act, r):
        """
        Add data point(s) and, if necessary, create the BO model.

        Parameters
        ---------
        obs: np.array
            Observation (n_data x obs_space_dim)
        act: np.array
            Action (n_data x obs_action_dim)
        r: float or np.array
            Reward
        """
        newX = np.atleast_2d(np.hstack((obs, act)))
        newY = np.atleast_2d(r)

        if self.BO is None:
            self.set_XY(newX, newY)  # Automatically creates BO
        else:
            oldX, oldY = self.get_XY()
            self.set_XY(np.vstack((oldX, newX)),
                        np.vstack((oldY, newY)))

    def set_XY(self, X, Y):
        """
        Set the values of the data points in the BO model.

        Since BO minimizes by default, we store -Y in case the optimize flag is
        set to True.

        Parameters
        ---------
        X: np.ndarray
            Array of inputs (n_data x input_dim)
        X: np.ndarray
            Array of observations (n_data x 1)
        """
        if X is not None and Y is not None:
            if self.maximize:
                Y = np.copy(Y)
                Y *= -1
            if self.BO is None:
                self._create_BO(X, Y)
            else:
                self.BO.X = X
                self.BO.Y = Y
        else:
            print('Skipping set XY step as the values provided are None')

    def _create_BO(self, X, Y):
        """
        Create the internal BO object with the values provided in the config.

        Together with the BO object, we create the exploit function and the
        exploit function optimizer that depend on it and are used to provide
        purely exploitative actions.

        Parameters
        ---------
        X: np.ndarray
            Array of inputs (n_data x input_dim)
        X: np.ndarray
            Array of observations (n_data x 1)
        """
        # Create BO
        self.BO = BayesianOptimization(f=None,
                                       domain=self.domain,
                                       model=self.model,
                                       acquisition_type=self.acquisition_type,
                                       model_update_interval=self.model_update_interval,
                                       normalize_Y=self.normalize_Y,
                                       X=X,
                                       Y=Y)
        # Create the internal GP model
        self.BO._update_model(self.BO.normalization_type)
        # The posterior mean is the function we optimize when we want to
        # exploit
        def exploit_f(x):
            x = np.atleast_2d(x)
            m, _ = self.predict(x, with_noise=False)
            if self.maximize:
                return -m
            else:
                return m

        self.exploit_func = exploit_f
        self.exploit_optimizer = AcquisitionOptimizer(self.BO.space, 'lbfgs')

    def predict(self, X, with_noise=False):
        """
        Predict mean and standard deviation of the reward at specified inputs.

        Parameters
        ----------
        X: np.ndarray
            Array of inputs where to make predictions (n_data x input_dim)
        with_noise: bool
            If true, predicts the function value + noise
        """
        if self.BO is not None:
            self.BO._update_model(self.BO.normalization_type)
            mu, sigma = self.model.predict(X, with_noise=with_noise)
            if self.maximize:
                mu *= -1
            return mu, sigma
        else:
            raise NotImplementedError('Not implemented the predict function '
                                      'when the BO model and its GP are not '
                                      'istanciated')

    def get_XY(self):
        """
        Get the input and observations.
        """
        if self.BO is not None:
            X = np.copy(self.BO.X)
            # We flip the sign in maximization case as BO stores -Y internally
            Y = -np.copy(self.BO.Y) if self.maximize else np.copy(self.BO.Y)
        else:
            X = Y = None
        return X, Y


class CustomGPModel(GPModel):
    """
    GP model that works with BO class with custom kern/lik.
    """
    def __init__(self, kernel, lik, num_inducing=10, sparse=False, **kwargs):
        super(CustomGPModel, self).__init__(**kwargs)
        # Overwrite kernel initialization of GPModel
        self.kernel = kernel
        self.lik = lik
        self.num_inducing = num_inducing
        self.kwargs = kwargs
        self.sparse = sparse

    def _create_model(self, X, Y):
        if not self.sparse:
            self.model = GPy.core.GP(X, Y, kernel=self.kernel,
                                     likelihood=self.lik)
        else:
            # Sample points from unit [0, 1]^d to reach the desired number
            # of points
            Z = np.random.rand(self.num_inducing, X.shape[1])

            self.model = GPy.core.SparseGP(X, Y, Z, kernel=self.kernel,
                                           likelihood=self.lik)

    def copy(self):
        copied_model = CustomGPModel(kernel=self.kernel,
                                     lik=self.lik,
                                     **self.kwargs)
        if self.model is not None:
            copied_model._create_model(self.model.X, self.model.Y)
            copied_model.updateModel(self.model.X, self.model.Y, None, None)
        return copied_model


class CustomGPModel_MCMC(GPModel_MCMC):
    """
    GP_MCMC model that works with BO class with custom kern/lik.
    """
    def __init__(self, kernel, lik, **kwargs):
        super(CustomGPModel_MCMC, self).__init__(**kwargs)
        # Overwrite kernel initialization of GPModel_MCMC
        self.kernel = kernel
        self.lik = lik
        self.kwargs = kwargs

    def _create_model(self, X, Y):
        self.model = GPy.core.GP(X, Y, kernel=self.kernel,
                                     likelihood=self.lik)

    def copy(self):
        copied_model = CustomGPModel_MCMC(kernel=self.kernel,
                                          lik=self.lik,
                                          **self.kwargs)
        if self.model is not None:
            copied_model._create_model(self.model.X, self.model.Y)
            copied_model.updateModel(self.model.X, self.model.Y, None, None)
        return copied_model


def space2domain(space, base_name=''):
    """Helper function to create GPyOpt domain from gym.space"""
    domain = []

    if isinstance(space, Box):
        if space.dtype in [np.int64, np.int32, np.int16]:
            vtype = 'discrete'
        elif space.dtype in [np.float64, np.float32, np.float16]:
            vtype = 'continuous'
        else:
            raise ValueError('Not supported dtype for conversion: {'
                             '}'.format(space.dtype))

        # GPyOpt cannot handle unbounded domains
        low = np.clip(space.low, -1e5, 1e5).flatten()
        high = np.clip(space.high, -1e5, 1e5).flatten()
        for i, (l, u) in enumerate(zip(low, high)):
            domain.append(
                {'name': '{}var_{}'.format(base_name, i),
                 'type': vtype,
                 'domain': (l, u),
                 'dimensionality': 1})
    elif isinstance(space, Discrete):
        domain.append({'name': '{}var_{}'.format(base_name, 0),
                       'type': 'discrete',
                       'domain': np.arange(space.n),
                       'dimensionality': 1})
    elif isinstance(space, MultiDiscrete):
        for i, n in enumerate(space.nvec):
            domain.append({'name': '{}var_{}'.format(base_name, i),
                           'type': 'discrete',
                           'domain': np.arange(n),
                           'dimensionality': 1})
    else:
        raise NotImplementedError('gym space to GPyOpt domain conversion '
                                  'implemented only for Box, Discrete and '
                                  'MultiDiscrete')
    return domain


def dict2ker(d):
    """
    Helper function to create kernel/likelihood from dictionary.

    Example dict structure
        d = {'type': RBF,
             'kwargs': {
                        'input_dim': {'v': 1},
                        'lengthscale': {'v': 0.1, 'prior': LogGaussian,
                                        'prior_params': {'mu': 0.1,
                                                         'sigma': 0.2},
                                        'constraint': 'bounded',
                                        'constraint_params': {'lower':1,
                                                             'upper':2}}
                        }}
    Should return
    k = GPy.kern.RBF(input_dim=1, lengthscale=0.1)
    k.lengthscale.set_prior(GPy.priors.LogGaussian(0.1, 0.2))
    k.lengthscale.constrain_bounded(1, 2)
    """
    # Initialize kernel
    tmp_dict = {param_name: param['v'] for param_name, param in d[
        'kwargs'].items()}
    k = d['type'](**tmp_dict)

    # Set priors and constraints
    for param_name, param in d['kwargs'].items():
        prior = param.get('prior')
        constraint = param.get('constraint')
        if prior is not None:
            prior_params = param.get('prior_params', {})
            if prior == GPy.priors.Gamma:
                k[param_name].set_prior(prior.from_EV(**prior_params))
            else:
                k[param_name].set_prior(prior(**prior_params))
        if constraint is not None:
            constraint_params = param.get('constraint_params')
            if constraint == 'bounded':
                k[param_name].constrain_bounded(**constraint_params)
            elif constraint == 'fixed':
                k[param_name].constrain_fixed(**constraint_params)
            elif constraint == 'negative':
                k[param_name].constrain_negative()
            elif constraint == 'positive':
                k[param_name].constrain_positive()
    return k
