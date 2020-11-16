import os
import numpy as np
import multiprocessing as mp
import uuid
from datetime import datetime

import src.envs.lunar_lander.utils as utils
from src.teacher.lunar_lander.teacher_env import create_single_switch_env, \
    create_teacher_env, evaluate_single_switch_policy, SingleSwitchPolicy
from src.envs.lunar_lander.interventions import LanderOrthogonalIntervention
from src.teacher.lunar_lander.analysis import plot_results, get_data_experiment_type


def evaluate_single_student(policy, base_dir=None, video=False,
                            teacher_env_kwargs={}):
    if base_dir is None:
        base_dir = os.path.join(os.path.abspath('.'), 'logs')
    exp_id = datetime.now().strftime('%d_%m_%y__%H_%M_%S') + str(uuid.uuid4())
    if teacher_env_kwargs['original']:
        name = 'original'
    else:
        name = policy.name
    logdir = os.path.join(base_dir, name, exp_id)
    os.makedirs(logdir, exist_ok=True)

    teacher_env, student_final_env_f = create_teacher_env(**teacher_env_kwargs)

    r, succ, crash, oom, to, tog, actions, failures = \
        evaluate_single_switch_policy(policy,
                                      teacher_env,
                                      student_final_env_f(),
                                      timesteps=int(1e5))

    np.savez(os.path.join(logdir, 'results.npz'), r=r, succ=succ,
             crash=crash, oom=oom, to=to, tog=tog, actions=actions,
             failures=failures)

    if video:
        env = utils.MonitorVideoIntervention(
            LanderOrthogonalIntervention(None, None, timeout=500),
            dirname=logdir, skipframe=10)
        obs = env.reset()
        for i in range(2000):
            a, _ = teacher_env.student.predict(obs)
            obs, r, g, done, info = env.step(a)
            if done:
                obs = env.reset()
        env.close()


def evaluate_parallel(policy_list, base_dir=None, teacher_env_kwargs={}):
    processes = []
    if base_dir is None:
        base_dir = os.path.join(os.path.abspath('.'), 'logs')
    for pi in policy_list:
        p = mp.Process(target=evaluate_single_student,
                       args=[pi, base_dir],
                       kwargs={'teacher_env_kwargs': teacher_env_kwargs})
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Need to load all the data and get the mean reward to pass back
    name = policy_list[0].name
    r, succ, crash, oom, to, tog, actions, failures = get_data_experiment_type(
        base_dir, name, return_mean=True)
    return np.mean(r)

