import numpy as np
import os
import time
from tabulate import tabulate
from src.teacher.lunar_lander.deploy_teacher_policy import \
    evaluate_parallel
from src.teacher.lunar_lander.teacher_env import SingleSwitchPolicy
from src.teacher.lunar_lander.analysis import get_data_experiment_type
import argparse


def run_comparision(log_dir, teacher_dir, teacher_env_kwargs={}):
    # Load teacher to get name
    teacher = SingleSwitchPolicy.load(os.path.join(
               teacher_dir, 'trained_teacher'))
    log_dir = os.path.join(log_dir, teacher.name)
    teacher_env_kwargs_copy = teacher_env_kwargs.copy()
    t = time.time()
    # Trained
    n_students = 10
    pi_list = [SingleSwitchPolicy.load(os.path.join(
               teacher_dir, 'trained_teacher'))] * n_students

    # Switch names to trained since the parameters are in the directory name
    for pi in pi_list:
        pi.name = 'trained'
    teacher_env_kwargs_copy.update({'original': False})
    evaluate_parallel(pi_list, base_dir=log_dir,
                      teacher_env_kwargs=teacher_env_kwargs_copy)

    # Wide
    pi_list = [SingleSwitchPolicy([-1000, 100],
                                  available_actions=[0, 1],
                                  name='wide')] * n_students
    evaluate_parallel(pi_list, base_dir=log_dir,
                      teacher_env_kwargs=teacher_env_kwargs_copy)
    # # Narrow
    pi_list = [SingleSwitchPolicy([400, 0],
                                  available_actions=[0, 1],
                                  name='narrow')] * n_students
    evaluate_parallel(pi_list, base_dir=log_dir,
                      teacher_env_kwargs=teacher_env_kwargs_copy)

    # Original
    pi_list = [SingleSwitchPolicy([400, 0],
                                  available_actions=[0, 1],
                                  name='original')] * n_students
    teacher_env_kwargs_copy.update({'original': True})
    evaluate_parallel(pi_list, base_dir=log_dir,
                      teacher_env_kwargs=teacher_env_kwargs_copy)
    print(f'elapsed {time.time() - t}')


def analyze_comparison(log_dir, teacher_dir):
    # Load teacher to get name
    teacher = SingleSwitchPolicy.load(os.path.join(
        teacher_dir, 'trained_teacher'))
    log_dir = os.path.join(log_dir, teacher.name)

    modes = ['trained', 'wide', 'narrow', 'original']
    metrics = ['succ', 'crash', 'oom', 'returns', 'training_failures']
    metric_summary = np.zeros((len(modes), len(metrics)), dtype=float)

    for i, name in enumerate(modes):
        r, succ, crash, oom, to, tog, actions, failures = \
        get_data_experiment_type(log_dir, name, return_mean=True)
        metric_summary[i][0] = np.mean(succ)
        metric_summary[i][1] = np.mean(crash)
        metric_summary[i][2] = np.mean(oom)
        metric_summary[i][3] = np.mean(r)
        metric_summary[i][4] = np.mean(failures)
        print(f'{name}-succ: {metric_summary[i, 0]}')
        print(f'{name}-crash: {metric_summary[i, 1]}')
        print(f'{name}-oom: {metric_summary[i, 2]}')
        print(f'{name}-r: {metric_summary[i, 3]}')
        print(f'{name}-failures: {metric_summary[i, 4]}')

    np.savez(os.path.join(log_dir, 'metrics_summary.npz'),
             metric_summary=metric_summary)


def get_metric_summary(log_dir, teacher_dir):
    teacher = SingleSwitchPolicy.load(os.path.join(teacher_dir,
                                                   'trained_teacher'))
    log_dir = os.path.join(log_dir, teacher.name)
    return np.load(os.path.join(log_dir, 'metrics_summary.npz'))['metric_summary']


def print_latex_table(mu, std):
    table = []
    for mu_row, std_row in zip(mu, std):
        line = []
        for j in range(len(mu_row)):
            line.append(f'${mu_row[j]:.3f}\pm{std_row[j]:.3f}$')
        table.append(line)
    print(tabulate(table, tablefmt="latex_raw"))


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               os.pardir, os.pardir, os.pardir, 'results',
                               'lunar_lander')
    base_teacher_dir = os.path.join(results_dir, 'teacher_training')

    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze', action='store_true', default=True,
                        help='Get the statistics of the comparison for the pre-trained teachers against the baselines')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Run the comparison between a pre-trained teacher and the baselines')
    parser.add_argument("--teacher_dir", nargs="*", type=str, default=[],
                        help='Directory(ies) containing the teacher to analyze or evaluate (assumed to be in result/flake/teacher_training)')
    parser.add_argument('scenario',
                        help="""
                        Choose among 3 available scenarios:
                        (0) student with two layers and noiseless observations
                        (1) student with one layer and noiseless observations
                        (2) student with two layers and noisy observations
                        """, type=int, default=0)
    args = parser.parse_args()

    if args.scenario == 0:
        n_layers = 2
        noise_std = 0.0
        B = 120
        time_steps_lim = int(1.5e6)
        log_dir = os.path.join(results_dir, 'teacher_comparison',
                               f'{n_layers}_layers_noise_{noise_std}')
    elif args.scenario == 1:
        n_layers = 1
        noise_std = 0.0
        B = 120
        time_steps_lim = int(1.5e6)
        log_dir = os.path.join(results_dir, 'teacher_comparison',
                               f'{n_layers}_layer_noise_{noise_std}')
    elif args.scenario == 2:
        n_layers = 2
        noise_std = 0.01
        B = 160
        time_steps_lim = int(2e6)
        log_dir = os.path.join(results_dir, 'teacher_comparison',
                               f'{n_layers}_layers_2milsteps_noise_{noise_std}')
    else:
        raise ValueError('Only three scenarios are available')

    sensor_noise = [noise_std ** 2] * 2 + [0] * 6
    teacher_env_kwargs = dict(sensor_noise=sensor_noise, n_layers=n_layers,
                              B=B, time_steps_lim=time_steps_lim)

    teachers = []
    for t in args.teacher_dir:
        if os.path.isdir(os.path.join(base_teacher_dir, t)):
            teachers.append(t)
        else:
            print(f'Could not find teacher {t} in {base_teacher_dir}')
    # Use default teacher is none is given
    if len(teachers) == 0:
        teachers = ['03_06_20__18_24_43']

    teachers_to_analyze = teachers if args.analyze else []
    teachers_to_run = teachers if args.evaluate else []

    for t in teachers_to_run:
        print(f'Evaluating teacher {t}')
        teacher_dir = os.path.join(base_teacher_dir, t)
        run_comparision(log_dir, teacher_dir, teacher_env_kwargs)

    for t in teachers_to_analyze:
        print(f'Analyzing teacher {t}')
        teacher_dir = os.path.join(base_teacher_dir, t)
        try:
            analyze_comparison(log_dir, teacher_dir)
        except FileNotFoundError:
            pass

    metrics_statistics = []
    for t in teachers_to_analyze:
        teacher_dir = os.path.join(base_teacher_dir, t)
        try:
            metrics_statistics.append(get_metric_summary(log_dir, teacher_dir))
        except FileNotFoundError:
            pass
    metrics_statistics = np.asarray(metrics_statistics)

    mu = metrics_statistics.mean(axis=0)
    std = metrics_statistics.std(axis=0) / np.sqrt(metrics_statistics.shape[0])
    print_latex_table(mu, std)
