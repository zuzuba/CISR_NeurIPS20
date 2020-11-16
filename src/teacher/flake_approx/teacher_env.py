import numpy as np

from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.a2c.utils import conv, linear, conv_to_fc


from src.envs import CMDP, FrozenLakeEnvCustomMap
from src.envs.frozen_lake.frozen_maps import MAPS
from src.students import LagrangianStudent, identity_transfer
from src.online_learning import ExponetiatedGradient
from src.teacher import FrozenLakeEvaluationLogger,  TeacherFrozenEnv, \
    create_intervention, SmallFrozenTeacherEnv
from src.teacher.frozen_lake_env import SmallFrozenTrainingObservation, SmallFrozenNonStationaryBandits
from src.envs.frozen_lake.utils import create_intervention_from_map, \
    OptimalAgent, add_teacher

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

__all__ = ['create_teacher_env', 'small_base_cenv_fn']


def constraint(info=None, **kwargs):
    return {'g': float(info['next_state_type'] in 'H')}


def small_base_env_fn():
    # Base MDP
    world_map = MAPS['small']
    not_slipping_prob = 0.8

    env_kwargs = dict(desc=world_map,
                      not_slipping_prob=not_slipping_prob,
                      base_r_mapping=None,
                      timeout=200)
    return FrozenLakeEnvCustomMap(**env_kwargs)


# Base CMDP
def small_base_cenv_fn():
    return CMDP(small_base_env_fn(), constraint,
                constraints_values=[0],
                n_constraints=1,
                avg_constraint=True)


def make_base_small_cenvs():
    # Base MDP
    world_map = MAPS['small']

    # # 2 interventions
    # dist = [1, 1]
    # tau = [0.1, 0]
    # buff_size = [1, 0]
    # avg_constraint = [True, True]

    # 3 Interventions
    dist = [2, 1, 1]
    tau = [0.1, 0.1, 0]
    buff_size = [1, 1, 0]
    avg_constraint = [True, True, True]
    interventions = []

    for d, t, b, avg in zip(dist, tau, buff_size, avg_constraint):
        interventions.append(
            create_intervention(
                small_base_cenv_fn,
                create_intervention_from_map(add_teacher(world_map, d)),
                [t], b, use_vec=True, avg_constraint=avg)
        )

    assert callable(interventions[0])
    test_env = create_intervention(
        small_base_cenv_fn(), create_intervention_from_map(add_teacher(
            world_map)),
        [0.0], 0, avg_constraint=True)

    return interventions, test_env


############################## TEACHER ENV ###################################


def my_small_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3,
                         stride=1, **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3,
                         stride=1, **kwargs))
    layer_3 = conv_to_fc(layer_2)
    return activ(
        linear(layer_3, 'fc1', n_hidden=32, init_scale=np.sqrt(2)))


def create_teacher_env(new_br_kwargs={}, new_online_kwargs={},
                       original=False, obs_from_training=False,
                       non_stationary_bandit=False):
    # Student definition
    br_kwargs = dict(policy=CnnPolicy, verbose=0, n_steps=128,
                     ent_coef=0.05, cliprange=0.2, learning_rate=1e-3,
                     noptepochs=9,
                     policy_kwargs={'cnn_extractor': my_small_cnn})
    br_kwargs.update(new_br_kwargs)

    # Define online kwargs
    online_kwargs = dict(B=0.5, eta=1.0)
    online_kwargs.update(new_online_kwargs)

    student_cls = LagrangianStudent
    n_envs = 4
    use_sub_proc_env = False
    student_default_kwargs = {'env': None,
                              'br_algo': PPO2,
                              'online_algo': ExponetiatedGradient,
                              'br_kwargs': br_kwargs,
                              'online_kwargs': online_kwargs,
                              'lagrangian_ronuds': 2,
                              'curriculum_transfer': identity_transfer,
                              'br_uses_vec_env': True,
                              'use_sub_proc_env': use_sub_proc_env,
                              'n_envs': n_envs,
                              }
    student_ranges_dict = {}

    # Teacher interventions
    if original:
        # To preserve the teacher env interface while training in the
        # original environment, we introduce a dummy intervention
        # condition that is always False.
        def dummy_intervention(**kwargs):
            return 0
        _, test_env = make_base_small_cenvs()
        intervention = create_intervention(
            base_cenv=small_base_cenv_fn,
            interventions=[dummy_intervention], taus=[0], buf_size=0,
            use_vec=True, avg_constraint=True)
        interventions = [intervention]
    else:
        interventions, test_env = make_base_small_cenvs()
    learning_steps = 4800 * 2
    time_steps_lim = learning_steps * 10
    test_episode_timeout = 200
    test_episode_number = 5
    if obs_from_training:
        env_cls = SmallFrozenTrainingObservation
    elif non_stationary_bandit:
        env_cls = SmallFrozenNonStationaryBandits
    else:
        env_cls = SmallFrozenTeacherEnv

    return env_cls(student_cls=student_cls,
                        student_default_kwargs=student_default_kwargs,
                        interventions=interventions,
                        final_env=test_env,
                        logger_cls=FrozenLakeEvaluationLogger,
                        student_ranges_dict=student_ranges_dict,
                        learning_steps=learning_steps,
                        test_episode_number=test_episode_number,
                        test_episode_timeout=test_episode_timeout,
                        time_steps_lim=time_steps_lim,
                        normalize_obs=False)
