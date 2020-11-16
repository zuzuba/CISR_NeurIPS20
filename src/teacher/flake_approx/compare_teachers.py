import numpy as np
import os
from functools import partial
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from tabulate import tabulate
import argparse

from src.teacher.flake_approx.deploy_teacher_policy import deploy_policy, \
    plot_deployment_metric, OpenLoopTeacher
from src.contextual_bandits import CGPUCBPolicy, ContextualBanditRL
from src.teacher.flake_approx.teacher_env import create_teacher_env, \
    small_base_cenv_fn
from src.teacher.frozen_single_switch_utils import SingleSwitchPolicy
from src.teacher.NonStationaryBanditPolicy import NonStationaryBanditPolicy
from src.utils.plotting import cm2inches, set_figure_params


def plot_comparison(log_dir, teacher_dir):
    text_width = cm2inches(13.968)  # Text width in cm
    figsize = (text_width / 2, text_width / 3.5)
    set_figure_params(fontsize=7)

    # Fix plotting when using command line on Mac
    plt.rcParams['pdf.fonttype'] = 42

    modes = ['Trained', 'SR1', 'SR2', 'HR', 'Original', 'Bandit']
    metric = ['successes', 'training_failures', 'averarge_returns']
    metric_summary = np.zeros((len(modes), len(metric)), dtype=float)
    teacher = SingleSwitchPolicy.load(os.path.join(teacher_dir,
                                                   'trained_teacher'))
    log_dir = os.path.join(log_dir, teacher.name)

    for i, subdir in enumerate(modes):
        if subdir == 'Trained':
            label = 'Optimized'
        elif subdir == 'Original':
            label = 'No interv.'
        else:
            label = subdir
        if os.path.isdir(os.path.join(log_dir, subdir)):
            for j, metric_name in enumerate(metric):
                fig = plt.figure(j, figsize=figsize)
                mu = plot_deployment_metric(os.path.join(log_dir, subdir),
                                            metric=metric_name, fig=fig,
                                            label=label, legend=True)
                metric_summary[i, j] = mu

    np.savez(os.path.join(log_dir, 'metrics_summary.npz'),
             metric_summary=metric_summary)
    for j, metric_name in enumerate(metric):
        plt.figure(j)
        plt.tight_layout(pad=0.2)
        plt.savefig(os.path.join(log_dir, metric_name + '.pdf'), format='pdf',
                    transparent=True)
        plt.close(j)


def run_comparision(log_dir, teacher_dir):
    env_f = partial(create_teacher_env)
    env_f_original = partial(create_teacher_env, original=True)
    env_f_single_switch = partial(create_teacher_env, obs_from_training=True)
    env_f_stationary_bandit = partial(create_teacher_env,
                                      non_stationary_bandit=True)
    teacher = SingleSwitchPolicy.load(os.path.join(teacher_dir,
                                                   'trained_teacher'))
    log_dir = os.path.join(log_dir, teacher.name)

    n_trials = 10
    t = time.time()
    for mode in ['Trained', 'SR1', 'SR2', 'HR', 'Original', 'Bandit']:
        if mode == 'SR2':
            model = OpenLoopTeacher([1])
        elif mode in ['SR1', 'Original']:
            model = OpenLoopTeacher([0])
        elif mode == 'HR':
            model = OpenLoopTeacher([2])
        elif mode == 'Bandit':
            model = NonStationaryBanditPolicy(3, 10)
        elif mode == 'Trained':
            model =SingleSwitchPolicy.load(os.path.join(teacher_dir,
                                           'trained_teacher'))
        processes = []

        for i in range(n_trials):
            log_tmp = os.path.join(log_dir, mode, f'experiment{i}')
            if mode == 'Original':
                teacher_env = env_f_original
            elif mode == 'Trained':
                teacher_env = env_f_single_switch
            elif mode == 'Bandit':
                teacher_env = env_f_stationary_bandit
            else:
                teacher_env = env_f
            p = mp.Process(target=deploy_policy,
                           args=[model, log_tmp, teacher_env,
                                 small_base_cenv_fn])
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    print(f'elapsed {time.time() - t}')


def run_bandits(log_dir):
    """
    Run n_rounds x n_trials students with bandit teacher from "Teacher-student
    curriculum learning" by Matiisen et al. and stores the results in log_dir.
    """
    env_f_stationary_bandit = partial(create_teacher_env,
                                      non_stationary_bandit=True)
    n_rounds = 3  # Number of times n_trials students are run
    n_trials = 10  # Number of students to run in parallel
    t = time.time()

    model = NonStationaryBanditPolicy(3, 10)

    for j in range(n_rounds):
        print(f'--------Running {j}th round of bandits----------')
        processes = []
        n_existing_experiments = len(os.listdir(log_dir))
        for i in range(n_trials):
            log_tmp = os.path.join(log_dir, f'experiment{i + n_existing_experiments}')
            teacher_env = env_f_stationary_bandit
            p = mp.Process(target=deploy_policy,
                           args=[model, log_tmp, teacher_env,
                                 small_base_cenv_fn])
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    print(f'elapsed {time.time() - t}')


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
                               'flake')
    log_dir = os.path.join(results_dir, 'teacher_comparison')
    base_teacher_dir = os.path.join(results_dir, 'teacher_training')

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Plot the comparison for the pre-trained teachers against the baselines')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Run the comparison between a pre-trained teacher and the baselines')
    parser.add_argument("--teacher_dir", nargs="*", type=str, default=[],
                        help='Directory(ies) containing the teacher to plot or evaluate (assumed to be in result/flake/teacher_training)')

    args = parser.parse_args()

    teachers = []
    for t in args.teacher_dir:
        if os.path.isdir(os.path.join(base_teacher_dir, t)):
            teachers.append(t)
        else:
            print(f'Could not find teacher {t} in {base_teacher_dir}')
    # Use default teacher is none is given
    if len(teachers) == 0:
        teachers = ['03_06_20__11_46_57']

    teachers_to_plot = teachers if args.plot else []
    teachers_to_run = teachers if args.evaluate else []

    for t in teachers_to_run:
        print(f'Evaluating teacher {t}')
        teacher_dir = os.path.join(base_teacher_dir, t)
        run_comparision(log_dir, teacher_dir)

    for t in teachers_to_plot:
        print(f'Plotting teacher {t}')
        teacher_dir = os.path.join(base_teacher_dir, t)
        plot_comparison(log_dir, teacher_dir)

    metrics_statistics = []
    for t in teachers_to_plot:
        teacher_dir = os.path.join(base_teacher_dir, t)
        metrics_statistics.append(get_metric_summary(log_dir, teacher_dir))
    metrics_statistics = np.asarray(metrics_statistics)

    # Print table
    mu = metrics_statistics.mean(axis=0)
    std = metrics_statistics.std(axis=0) / np.sqrt(metrics_statistics.shape[0])
    print_latex_table(mu, std)
