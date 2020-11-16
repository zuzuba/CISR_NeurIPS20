import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
import time
from itertools import cycle
from functools import partial


from src.teacher.flake_approx.teacher_env import create_teacher_env
from src.envs.frozen_lake.utils import plot_trajectories, deploy, plot_networks

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

__all__ = ['deploy_policy', 'plot_deployment_metric', 'OpenLoopTeacher']


def deploy_policy(policy, log_dir, env_f, deployment_env_fn=None):
    os.makedirs(log_dir, exist_ok=True)
    teacher_env = env_f()
    obs_t = teacher_env.reset()
    student = teacher_env.student

    n_steps = int(teacher_env.time_steps_lim / teacher_env.learning_steps) + 1
    successes = np.zeros(n_steps, dtype=float)
    training_failures = np.zeros(n_steps, dtype=float)
    averarge_returns = np.zeros(n_steps, dtype=float)
    teacher_rewards = np.zeros(n_steps, dtype=float)
    teacher_observations = np.zeros((obs_t.size, n_steps), dtype=float)

    i = 0
    while True:
        a, _ = policy.predict(obs_t)
        obs_t, teacher_rewards[i], done, _ = teacher_env.step(a)
        if hasattr(policy, 'add_point'):  # For non stationary bandit policy
            policy.add_point(a, teacher_rewards[i])
        teacher_observations[:, i] = obs_t
        if deployment_env_fn is None:
            env = teacher_env.final_env
        else:
            env = deployment_env_fn()
        # env = teacher_env.actions[a]()
        succ, avg_r, avg_r_succ, traj = deploy(student, env, 10000)
        successes[i], averarge_returns[i] = succ, avg_r
        training_failures[i] = teacher_env.student_failures

        print(f'{i+1} training-> success: {succ}\tavg_ret: '
              f'{avg_r}\tavg_ret_succ {avg_r_succ}\t action {a}')
        plot_trajectories(traj, env.desc.shape)
        plt.savefig(os.path.join(log_dir, f'trajectories{i}.pdf'),format='pdf')
        # plot_networks(student.br, env.desc.shape)
        plt.close()
        i += 1
        if done:
            break

    # Plot successes and returns
    np.savez(os.path.join(log_dir, 'results.npz'),
             successes=successes, averarge_returns=averarge_returns,
             teacher_rewards=teacher_rewards, training_failures=training_failures)
    plt.figure()
    f, axes = plt.subplots(1, 3)
    metrics = [successes, averarge_returns, teacher_rewards]
    titles = ['successes', 'student ret', 'teacher ret']
    for a, metric, title in zip(axes, metrics, titles):
        if title == 'teacher ret':
            a.plot(np.cumsum(metric))
        else:
            a.plot(metric)
        a.set_title(title)

    plt.savefig(os.path.join(log_dir, 'results.pdf'), format='pdf')

    plt.figure()
    for o_num, o in enumerate(teacher_observations):
        plt.plot(o, label=o_num)
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'teacher_observations.pdf'),
                format='pdf')


def plot_deployment_metric(log_dir, metric, ax=None, fig=None, label=None, legend=True):
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.gca()
    returns = []
    for subdir in os.listdir(log_dir):
        if os.path.isdir(os.path.join(log_dir, subdir)):
            try:
                data = np.load(os.path.join(log_dir, subdir, 'results.npz'))
                ret = data[metric]
                returns.append(ret)
            except FileNotFoundError:
                    pass
    if metric == 'teacher_rewards':
        returns = np.cumsum(np.asarray(returns)[:, :], axis=1)
    else:
        returns = np.asarray(returns)
    mu, std = np.mean(returns, axis=0), np.std(returns, axis=0)
    print(f'{log_dir.split("/")[-1]} - {metric} final mean -> {mu[-1]}')
    std = std / np.sqrt(returns.shape[0])
    if label is None:
        label = log_dir.split('/')[-1].replace('_', ' ')
    ax.plot(mu, label=label)
    ax.fill_between(np.arange(mu.size), mu-std, mu+std, alpha=0.5)
    if legend:
        plt.legend(frameon=False, loc='best')
    # Ticks
    plt.tick_params(axis='both',
                    which='both',
                    bottom=True,
                    left=True)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('Curriculum steps')
    plt.ylabel(metric.replace('_', ' '))
    plt.tight_layout(pad=0.2)

    return mu [-1]


class OpenLoopTeacher(object):
    """
    Dummy teacher that cycles through a sequence of actions
    """
    def __init__(self, action_sequence):
        self.actions = cycle(action_sequence)

    def predict(self, obs):
        return next(self.actions), None


if __name__ == '__main__':
    n = 5
    small_map = True
    experiment_name = 'cnn_small_env_3_interventions'
    actions_sequences = [[0] * 11,
                         [1] * 6 + [2] * 5]

    policy_names = ['zeros', '1x6_2x5']
    env_f = partial(create_teacher_env)
    t = time.time()
    for actions, pi_name in zip(actions_sequences, policy_names):
        processes = []
        for i in range(n):
            name = os.path.join(os.path.abspath('.'), experiment_name,
                                pi_name, f'experiment_{i}')
            pi = OpenLoopTeacher(actions)
            p = mp.Process(target=deploy_policy, args=[pi, name, env_f])
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    print(f'Elapsed {time.time() - t}')

    # Plot successes and teacher reward

    plt.figure()
    exp_dir = os.path.join(os.path.abspath('.'), experiment_name)
    for subdir in os.listdir(exp_dir):
        if os.path.isdir(os.path.join(exp_dir, subdir)):
             plot_deployment_metric(os.path.join(exp_dir, subdir),
                                    metric='teacher_rewards')
    plt.savefig(os.path.join(os.path.abspath('.'), experiment_name,
                             'policy_copmarison.pdf'), format='pdf')

    plt.figure()
    exp_dir = os.path.join(os.path.abspath('.'), experiment_name)
    for subdir in os.listdir(exp_dir):
        if os.path.isdir(os.path.join(exp_dir, subdir)):
            plot_deployment_metric(os.path.join(exp_dir, subdir),
                                   metric='successes')
    plt.savefig(os.path.join(os.path.abspath('.'), experiment_name,
                             'successes.pdf'), format='pdf')
