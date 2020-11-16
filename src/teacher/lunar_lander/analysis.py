import os
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['get_data_experiment_type', 'plot_results']


def get_data_experiment_type(base_dir, exp_name, return_mean=True):
    exp_group_dir = os.path.join(base_dir, exp_name)
    r, succ, crash, oom, to, tog, actions, failures = [], [], [], [], [], \
                                                      [], [], []

    for single_exp in os.listdir(exp_group_dir):
        single_exp_dir = os.path.join(exp_group_dir, single_exp)
        fname = os.path.join(single_exp_dir, 'results.npz')
        if os.path.isfile(fname):
            data = np.load(fname)
            if return_mean:
                r.append(data['r'].mean())
                succ.append(data['succ'].mean())
                crash.append(data['crash'].mean())
                oom.append(data['oom'].mean())
                to.append(data['to'].mean())
                tog.append(data['tog'].mean())
            else:
                r.append(data['r'])
                succ.append(data['succ'])
                crash.append(data['crash'])
                oom.append(data['oom'])
                to.append(data['to'])
                tog.append(data['tog'])
            actions.append(data['actions'])
            failures.append((data['failures']))
    return r, succ, crash, oom, to, tog, actions, failures


def plot_results(log_dir):
    r_list = []
    succ_list = []
    crash_list = []
    oom_list = []
    to_list = []
    tog_list = []
    actions_list = []
    failures_list = []
    name_list = []
    for subdir in os.listdir(log_dir):
        exp_dir = os.path.join(log_dir, subdir)
        if os.path.isdir(exp_dir):
            name_list.append(subdir)
            r, succ, crash, oom, to, tog, actions, failures = \
                get_data_experiment_type(
                log_dir, subdir, return_mean=True)
            r_list.append(r)
            succ_list.append(succ)
            crash_list.append(crash)
            oom_list.append(oom)
            to_list.append(to)
            tog_list.append(tog)
            actions_list.append(actions)
            failures_list.append(failures)
    values_list = [r_list, succ_list, crash_list, oom_list, to_list, tog_list,
                   failures_list]
    names_list = ['r', 'succ', 'crash', 'oom', 'to',  'tog', 'failures']

    for metric_value, metric_name in zip(values_list, names_list):
        plt.figure()
        for i, name in enumerate(name_list):
            try:
                v = np.hstack(metric_value[i])
            except ValueError:
                v = metric_value[i]
            plt.scatter(np.full_like(v, i), v, label=name)
            plt.scatter(i, np.mean(v), s=100)
            plt.title(metric_name)
            print(f'{name}-{metric_name}: {np.mean(v)}')

    for actions in actions_list:
        plt.figure()
        actions = np.asarray(actions)
        plt.plot(actions.T)
    plt.legend()
    plt.show()