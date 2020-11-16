import numpy as np
import matplotlib.pyplot as plt
import src.envs.lunar_lander.utils as utils
import src.envs.lunar_lander.conditions as conditions

# from stable_baselines import PPO2
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import SubprocVecEnv
# from stable_baselines.common.evaluation import evaluate_policy

__all__ = ['plot_funnel', 'plot_value']


def plot_funnel(ax, coef, threshold=None, c=None, label=None, lw=1):
    if c is None:
        c = 'tab:green'
    threshold = -1000 if threshold is None else threshold

    xtmp = np.linspace(-1, utils.l_heli, 50)
    ytmp = np.zeros_like(xtmp)
    for i, xi in enumerate(xtmp):
        ytmp[i] = np.maximum(utils.y_envelop(xi, coef), threshold)
    if label is None:
        ax.plot(xtmp, ytmp, c=c, lw=lw)
    else:
        ax.plot(xtmp, ytmp, c=c, label=label, lw=lw)

    xtmp = np.linspace(utils.r_heli, 1, 50)
    ytmp = np.zeros_like(xtmp)
    for i, xi in enumerate(xtmp):
        ytmp[i] = np.maximum(utils.y_envelop(xi, coef), threshold)
    ax.plot(xtmp, ytmp, c=c, lw=lw)


def plot_value(agent, env=None, filename=None):
    x_velocities = [-0.1, 0.1]
    y_velocities = [-0.1, 0.1]
    angle = [-np.pi/20, np.pi/20]
    ang_velocities = [-0.05, 0.05]
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-0.2, 1.3, 100)
    xx, yy = [v.ravel() for v in np.meshgrid(x, y)]
    shape = (len(x), len(y))

    fig, ax = plt.subplots(4, 4, figsize=(20, 20))

    i = 0
    for x_vel in x_velocities:
        for y_vel in y_velocities:
            j = 0
            for ang in angle:
                for ang_vel in ang_velocities:
                    s = np.column_stack((xx,
                                         yy,
                                         np.full_like(xx, x_vel),
                                         np.full_like(xx, y_vel),
                                         np.full_like(xx, ang),
                                         np.full_like(xx, ang_vel),
                                         np.full_like(xx, 0),
                                         np.full_like(xx, 0)))
                    value = agent.train_model.value(s).reshape(shape)
                    ax[i, j].imshow(value, origin='lower',
                                    extent=(x[0], x[-1], y[0], y[-1]))
                    if env is not None:
                        plot_funnel(ax[i, j], env)
                    ax[i, j].set_title(f'xvel {x_vel:.2f} yvel {y_vel:.2f} ang {ang:.2f} '
                                       f'ang_vel {ang_vel:.2f}')
                    ax[i, j].set_xlim(x[0], x[-1])
                    ax[i, j].set_ylim(y[0], y[-1])

                    j += 1
            i += 1
    if filename is not None:
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        filename = os.path.join(os.path.abspath('.'), filename)
        print(filename)
        plt.savefig(filename, format='pdf')
    plt.show()


# def main(load=False):
#     # Define env
#     stabilize_penalty = partial(stabilize, penalty=15)
#     env_f = lambda : LunarLanderCustomBuffer(history_length=20,
#                                              condition=[MinimalCondition(
#                                                  coef=12)],
#                                              modification=stabilize_penalty,
#                                              timeout=500)
#     agent_file = os.path.join(os.path.abspath('.'),
#                               'trained_agent_for_plotting')
#     # Load
#     if load:
#         agent = PPO2.load(agent_file)
#
#     # Otherwise learn
#     else:
#         vec_env = SubprocVecEnv([env_f] * 4)
#         agent = PPO2(MlpPolicy, vec_env, ent_coef=1e-3,
#                               learning_rate=5e-3,
#                      policy_kwargs={'net_arch': [16, 16]}, n_steps=500,
#                      noptepochs=32)
#         agent.learn(int(5e5))
#         agent.save(agent_file)
#
#         # Get video of learned agent
#         video_dir = os.path.join(os.path.abspath('.'), 'videos',
#                                  'value_function_plot')
#         os.makedirs(video_dir, exist_ok=True)
#         env = CustomMonitorVideo(env_f(), dirname=video_dir)
#         mu, _ = evaluate_policy(agent, env)
#         print(f'Average reward {mu}')
#         env.close()
#
#     # Plot value function and funnel
#     plot_value(agent, env_f(), 'final_value')
#
#
# def transfer_to_middle():
#     agent_file = os.path.join(os.path.abspath('.'),
#                               'trained_agent_for_plotting')
#
#     stabilize_penalty = partial(stabilize, penalty=20)
#     env_f = lambda: LunarLanderCustomBuffer(history_length=75,
#                                             condition=[MinimalCondition(
#                                                 coef=1)],
#                                             modification=stabilize_penalty,
#                                             timeout=500)
#     vec_env = SubprocVecEnv([env_f] * 4)
#     agent = PPO2.load(agent_file, env=vec_env)
#
#     # Get video of learned agent
#     video_dir = os.path.join(os.path.abspath('.'), 'videos',
#                              'after_transfer_before_learning')
#     os.makedirs(video_dir, exist_ok=True)
#     env = CustomMonitorVideo(env_f(), dirname=video_dir)
#     mu, _ = evaluate_policy(agent, env)
#     print(f'Average reward {mu}')
#     env.close()


if __name__ == '__main__':
    # main()
    # transfer_to_middle()
    pass