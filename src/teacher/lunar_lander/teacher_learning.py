import numpy as np
import os
import GPy
from GPyOpt.methods import BayesianOptimization
from GPyOpt.models import GPModel, GPModel_MCMC
from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer
import time
from datetime import datetime
from src.teacher.lunar_lander.deploy_teacher_policy import evaluate_parallel
from src.teacher.lunar_lander.teacher_env import SingleSwitchPolicy


def main():
    domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (-200, 200)},
              {'name': 'var_2', 'type': 'continuous', 'domain': (0, 6)},
              {'name': 'var_3', 'type': 'discrete', 'domain': (0, 1)},
              {'name': 'var_4', 'type': 'discrete', 'domain': (0, 1)}]
    kern = GPy.kern.RBF(input_dim=4, variance=1, lengthscale=[20, 1, 0.1, 0.1],
                        ARD=True)
    kern.lengthscale.priors.add(GPy.priors.Gamma.from_EV(20, 4), np.array([0]))
    kern.lengthscale.priors.add(GPy.priors.Gamma.from_EV(1, 0.3), np.array([1]))
    kern.lengthscale.priors.add(GPy.priors.Gamma.from_EV(0.2, 0.2),
                                np.array([2, 3]))
    kern.variance.set_prior(GPy.priors.Gamma.from_EV(1, 0.2))

    model = GPModel(kernel=kern, noise_var=0.01, max_iters=1000)
    # bo_objective = lambda x: x[0, 0] * x[0, 2] + x[0, 1] * x[0, 3]

    exp_starting_time = datetime.now().strftime('%d_%m_%y__%H_%M_%S')
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, os.pardir, os.pardir, 'results',
                            'lunar_lander', 'teacher_training',
                            exp_starting_time)
    os.makedirs(base_dir, exist_ok=True)

    def init_teaching_policy(params, name=None):
        params = np.squeeze(np.array(params))
        params = np.copy(params)
        thresholds = params[:2]
        available_actions = params[2:].astype(np.int64)
        policy = SingleSwitchPolicy(thresholds, available_actions)
        return policy

    def bo_objective(params):
        teacher_env_kwargs = dict(sensor_noise=[0.0] * 8, n_layers=2,
                                  B=120, time_steps_lim=int(1.5e6),
                                  original=False)
        policy_list = [init_teaching_policy(params) for _ in range(10)]
        return evaluate_parallel(policy_list, base_dir=base_dir,
                                 teacher_env_kwargs=teacher_env_kwargs)

    # Initialize with one value per configuration
    initial_X = np.array([[0, 3, 0, 0],
                          [0, 3, 0, 1],
                          [0, 3, 1, 0],
                          [0, 3, 1, 1]])

    my_bo = BayesianOptimization(bo_objective,
                                 domain=domain,
                                 initial_design_numdata=0,
                                 initial_design_type='random',
                                 acquisition_type='LCB',
                                 maximize=True,
                                 normalize_Y=True,
                                 model_update_interval=1,
                                 X=initial_X,
                                 model=model)
    my_bo.suggest_next_locations()  # Creates the GP model
    my_bo.model.model['Gaussian_noise.variance'].set_prior(
        GPy.priors.Gamma.from_EV(0.01, 0.1))

    t = time.time()
    my_bo.run_optimization(10,
                           report_file=os.path.join(base_dir, 'bo_report.txt'),
                           evaluations_file=os.path.join(base_dir,
                                                         'bo_evaluations.csv'),
                           models_file=os.path.join(base_dir, 'bo_model.csv'),
                           verbosity=True)
    print(f'Optimization complete in {time.time() - t}')
    print(f'Policy with optimal observation: {my_bo.x_opt}')
    print(f'Value of the optimal observation: {my_bo.fx_opt}')

    np.savez(os.path.join(base_dir, 'solution.npz'), xopt=my_bo.x_opt,
             fxopt=my_bo.fx_opt, X=my_bo.X, Y=my_bo.Y)
    trained_policy = init_teaching_policy(my_bo.x_opt)
    save_path = os.path.join(base_dir, 'trained_teacher')
    trained_policy.save(save_path)


if __name__ == '__main__':
    main()
