import numpy as np
import os
import GPy
from GPyOpt.methods import BayesianOptimization
from GPyOpt.models import GPModel
import time
from datetime import datetime
from src.teacher.frozen_single_switch_utils import evaluate_single_switch_policy, \
    SingleSwitchPolicy
from src.teacher.flake_approx.teacher_env import create_teacher_env, \
    small_base_cenv_fn


def main(n_interv=3):
    if n_interv == 2:
        domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, 6)},
                  {'name': 'var_2', 'type': 'continuous', 'domain': (0, 0.5)}]
        kern = GPy.kern.RBF(input_dim=2, variance=1, lengthscale=[1., 0.05],
                            ARD=True)
        model = GPModel(kernel=kern, noise_var=0.1, max_iters=0)

        teacher_env = create_teacher_env(obs_from_training=True)
        student_final_env = small_base_cenv_fn()

        def bo_objective(thresholds):
            thresholds = np.array(thresholds)
            if thresholds.ndim == 2:
                thresholds = thresholds[0]
            policy = SingleSwitchPolicy(thresholds)
            return evaluate_single_switch_policy(policy, teacher_env, student_final_env)
    elif n_interv == 3:
        domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (-0.5,
                                                                     5.5)},
                  {'name': 'var_2', 'type': 'continuous', 'domain': (0, 0.2)},
                  {'name': 'var_3', 'type': 'continuous', 'domain': (-0.5,
                                                                     5.5)},
                  {'name': 'var_4', 'type': 'continuous', 'domain': (0, 0.2)},
                  {'name': 'var_5', 'type': 'discrete', 'domain': (0, 1, 2)},
                  {'name': 'var_6', 'type': 'discrete', 'domain': (0, 1, 2)},
                  {'name': 'var_7', 'type': 'discrete', 'domain': (0, 1, 2)}]

        kern = GPy.kern.RBF(input_dim=7, variance=1,
                            lengthscale=[1., 0.05, 1, 0.05, 0.5, 0.5, 0.5],
                            ARD=True)
        kern.lengthscale.priors.add(GPy.priors.Gamma.from_EV(1, 1),
                                    np.array([0, 2]))
        kern.lengthscale.priors.add(GPy.priors.Gamma.from_EV(0.05, 0.02),
                                    np.array([1, 3]))
        kern.lengthscale.priors.add(GPy.priors.Gamma.from_EV(0.2, 0.2),
                                    np.array([4, 5, 6]))
        kern.variance.set_prior(GPy.priors.Gamma.from_EV(1, 0.2))
        model = GPModel(kernel=kern, noise_var=0.05, max_iters=1000)

        teacher_env = create_teacher_env(obs_from_training=True)
        student_final_env = small_base_cenv_fn()

        def init_teaching_policy(params, name=None):
            params = np.squeeze(np.array(params))
            thresholds = params[:4]
            thresholds = thresholds.reshape(2, 2)
            available_actions = params[4:].astype(np.int64)
            policy = SingleSwitchPolicy(thresholds, available_actions, name=name)
            return policy

        def bo_objective(params):
            policy = init_teaching_policy(params)
            return evaluate_single_switch_policy(policy, teacher_env,
                                                 student_final_env)

    # Logging dir
    exp_starting_time = datetime.now().strftime('%d_%m_%y__%H_%M_%S')
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               os.pardir, os.pardir, os.pardir, 'results',
                               'flake')
    base_dir = os.path.join(results_dir, 'teacher_training', exp_starting_time)
    os.makedirs(base_dir, exist_ok=True)

    my_bo = BayesianOptimization(bo_objective,
                                 domain=domain,
                                 initial_design_numdata=10,
                                 initial_design_type='random',
                                 acquisition_type='LCB',
                                 maximize=True,
                                 normalize_Y=True,
                                 model_update_interval=1,
                                 model=model)

    my_bo.suggest_next_locations()  # Creates the GP model
    my_bo.model.model['Gaussian_noise.variance'].set_prior(
        GPy.priors.Gamma.from_EV(0.01, 0.1))

    t = time.time()
    my_bo.run_optimization(20,
                           report_file=os.path.join(base_dir, 'bo_report.txt'),
                           evaluations_file=os.path.join(base_dir,
                                                         'bo_evaluations.csv'),
                           models_file=os.path.join(base_dir, 'bo_model.csv'))
    print(f'Optimization complete in {time.time() - t}')
    print(f'Optimal threshold: {my_bo.x_opt}')
    print(f'Optimal return: {my_bo.fx_opt}')
    np.savez(os.path.join(base_dir, 'solution.npz'), xopt=my_bo.x_opt,
             fxopt=my_bo.fx_opt)
    trained_policy = init_teaching_policy(my_bo.x_opt)
    save_path = os.path.join(base_dir, 'trained_teacher')
    trained_policy.save(save_path)


if __name__ == '__main__':
    main()
