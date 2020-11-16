import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import time
from gym.core import Wrapper

try:
    from stable_baselines import bench
except ImportError:
    bench = None

import src.envs.lunar_lander.plotting as plotting
import src.envs.lunar_lander.conditions as conditions
import src.envs.lunar_lander.modifications as modifications
from src.utils.plotting import set_figure_params, hide_all_ticks, cm2inches

from gym.envs.box2d.lunar_lander import VIEWPORT_W, \
    VIEWPORT_H, SCALE, LEG_DOWN, FPS


__all__ = ['absolute_coord_to_landing_coord',
           'landing_coord_to_absolute_coord',
           'get_height_function',
           'l_heli', 'r_heli',
           'CustomMonitorVideo',
           'CustomMonitorPerformance',
           'inside_helipad',
           'y_envelop']


def absolute_coord_to_landing_coord(x, y):
    """
    Go from absolute position to coordinate system of lunar lander
    observations (centered at landing spot and scaled).
    """
    newx = (x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2)
    H = VIEWPORT_H / SCALE
    helipad_y = H / 4
    newy = (y - (helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2)
    return newx, newy


def landing_coord_to_absolute_coord(x, y):
    """
    Go from landing coordinate to the absolute reference system of the
    plotting window.
    """
    newx = x * (VIEWPORT_W/SCALE/2) + VIEWPORT_W/SCALE/2
    H = VIEWPORT_H / SCALE
    helipad_y = H / 4
    newy = y * (VIEWPORT_H/SCALE/2) + (helipad_y+LEG_DOWN/SCALE)
    return newx, newy


def get_height_function(env):
    """
    Find callable that, for given x value, returns the corresponding y of
    the moon surface for the given env.
    """
    x = [env.sky_polys[0][0][0]]
    x.extend([v[1][0] for v in env.sky_polys])
    y = [env.sky_polys[0][0][1]]
    y.extend([v[1][1] for v in env.sky_polys])
    scaled_x, scaled_y = absolute_coord_to_landing_coord(np.asarray(x),
                                                         np.asarray(y))

    def height_f(xpos):
        return np.interp(xpos, scaled_x, scaled_y)
    return height_f


if bench is not None:
    class CustomMonitorPerformance(bench.Monitor):
        """
        Monitor for lunar lander that keeps track of successes, crashes out of map
        events, timeouts, timeouts on ground and interventions besides returns.
        """
        def __init__(self, env, filename, allow_early_resets=True,
                     reset_keywords=(), info_keywords=()):
            super(CustomMonitorPerformance, self).__init__(
                env, filename, allow_early_resets, reset_keywords, info_keywords)
            self.crashes = []
            self.successes = []
            self.oom = []
            self.timeouts = []
            self.timeouts_on_ground = []
            self.interventions = []
            self.ep_interventions = 0

        def step(self, action):
            obs, r, done, info = super().step(action)
            self.ep_interventions += info['intervention']
            if done:
                self.successes.append(not self.env.lander.awake)
                self.crashes.append(self.env.game_over)
                self.oom.append(abs(obs[0]) >= 1.0)
                timeout = self.env.num_steps >= self.env.timeout
                self.timeouts.append(timeout)
                self.timeouts_on_ground.append(timeout and np.any(obs[-2:] == 1))
                self.interventions.append(self.ep_interventions)
                self.ep_interventions = 0
            return obs, r, done, info

        def get_episode_successes(self):
            return self.successes

        def get_episode_crashes(self):
            return self.crashes

        def get_episode_oom(self):
            return self.oom

        def get_episode_timeouts(self):
            return self.timeouts

        def get_episode_timeouts_on_ground(self):
            return self.timeouts_on_ground

        def get_episode_interventions(self):
            return self.interventions

        def reset(self, **kwargs):
            self.ep_interventions = 0
            return super().reset(**kwargs)


    class MonitorPerformanceIntervention(CustomMonitorPerformance):
        def __init__(self, env, filename, allow_early_resets=True,
                     reset_keywords=(), info_keywords=()):
            super().__init__(env, filename, allow_early_resets, reset_keywords,
                             info_keywords)
            # We distinguish between the intervention wrapper (called for step)
            # from the basic lunar lander env (called for lander position and so
            # on)
            self.env
            self.unconstrained_env = self.env.env

        def step(self, action):
            if self.needs_reset:
                raise RuntimeError(
                    "Tried to step environment that needs reset")
            obs, r, g, done, info = self.env.step(action)

            self.rewards.append(r)
            self.ep_interventions += info['teacher_intervention']
            if done:
                # Part from original stable_baselines monitor that we rewrite
                # because we cannot  call super().step due to the constraint.
                self.needs_reset = True
                ep_rew = sum(self.rewards)
                eplen = len(self.rewards)
                self.episode_rewards.append(ep_rew)
                self.episode_lengths.append(eplen)
                self.episode_times.append(time.time() - self.t_start)
                self.total_steps += 1

                # Part of the monitor specific to lunar lander we rewrite due to
                # the difference between env and modified_env
                self.successes.append(not self.unconstrained_env.lander.awake)
                self.crashes.append(self.unconstrained_env.game_over)
                self.oom.append(abs(obs[0]) >= 1.0)
                timeout = self.env.num_steps >= self.env.timeout
                self.timeouts.append(timeout)
                self.timeouts_on_ground.append(
                    timeout and np.any(obs[-2:] == 1))
                self.interventions.append(self.ep_interventions)
                self.ep_interventions = 0
            return obs, r, g, done, info

        def reset(self, **kwargs):
            self.ep_interventions = 0

            # Part copied from the stable baselines monitor because we can't
            # call super().reset() due to the difference in env naming. If
            # we do, we are only going to reset the underlying lunar env and
            # not the intervention (therefore we do not reset the counter
            # for the timeout),
            if not self.allow_early_resets and not self.needs_reset:
                raise RuntimeError(
                    "Tried to reset an environment before done. If you want to allow early resets, "
                    "wrap your env with Monitor(env, path, allow_early_resets=True)")
            self.rewards = []
            self.needs_reset = False
            for key in self.reset_keywords:
                value = kwargs.get(key)
                if value is None:
                    raise ValueError(
                        'Expected you to pass kwarg %s into reset' % key)
                self.current_reset_info[key] = value
            return self.env.reset(**kwargs)
else:
    CustomMonitorPerformance = None
    MonitorPerformanceIntervention = None


class CustomMonitorVideo(Wrapper):
    """
    Monitor to take video of lunar lander episodes without using rendering.
    This is done by saving a bunch of plots and by turning them into a video
    with ffmpeg.
    """
    def __init__(self, env, dirname=None, skipframe=10, paper=False,
                 presentation_video=False):
        """
        Parameters
        ----------
        env: LunarLander env
        dirname: str
            Name of the folder where to save the video
        skipframe: int
            We plot every skipframe frames for efficiency
        paper: bool
            Whether to produce the plots for the paper
        """
        self.height_f = None
        self.x = np.linspace(-1, 1, 100)
        self.y = np.zeros_like(self.x)

        self.dirname = dirname
        self.paper = paper
        set_figure_params(fontsize=4.5)
        w = cm2inches(0.36*14/2)
        h = w * 1.75/2
        if presentation_video:
            set_figure_params(fontsize=12)
            w = cm2inches(10)
            h = w * 1.75 / 2
        self.fig, self.ax = plt.subplots(figsize=(w, h))
        super(CustomMonitorVideo, self).__init__(env)
        self.episode_steps = 0
        self.n_figures = 0
        self.skipframe = skipframe
        self.presentation_video = presentation_video
        try:
            if isinstance(self.env, CustomMonitorPerformance):
                self.env = self.env.env
        except TypeError as e:
            print(e)

        if hasattr(self.env, 'condition'):
            self.condition = self.env.condition
        else:
            self.condition = []

        if hasattr(self.env, 'modification'):
            self.modification = self.env.modification
        else:
            self.condition = []

    def get_background(self):
        """
        Get x and y of moon surface.
        """
        self.height_f = get_height_function(self.env)
        self.y = self.height_f(self.x)

    def _plot(self, plot_lander=True):
        self.ax.clear()
        # self.ax.plot(self.x, self.y)
        self.ax.fill_between(self.x, self.y, np.full_like(self.x, 2),
                             color=(0, 0, 0))

        # Plot lander
        if plot_lander:
            lander_x, lander_y = \
                absolute_coord_to_landing_coord(
                self.env.lander.position.x, self.env.lander.position.y)
            if self.presentation_video:
                self.ax.scatter(lander_x, lander_y, s=200,
                                c=(0.5, 0.4, 0.9))
            else:
                self.ax.scatter(lander_x, lander_y, s=10, label='Lander',
                                c=(0.5, 0.4, 0.9))

            # Plot legs
            for l in self.env.legs:
                leg_x, leg_y = absolute_coord_to_landing_coord(
                    l.position.x, l.position.y)
                if self.presentation_video:
                    self.ax.scatter(leg_x, leg_y, s=40, c=(0.5, 0.4, 0.9))
                else:
                    self.ax.scatter(leg_x, leg_y, s=2, c=(0.5, 0.4, 0.9))

        self.ax.set_xlim([-1, 1])
        _, ylim = absolute_coord_to_landing_coord(
            0, np.array([0, VIEWPORT_H / SCALE]))
        self.ax.set_ylim(ylim)

        for c in self.condition:
            if self.presentation_video:
                lw = 4
            else:
                lw = 1
            if isinstance(c, conditions.MinimalCondition):
                plotting.plot_funnel(self.ax, c.coef, c.highest_y,
                                     c='tab:green', label='Trigger',lw=lw)
            if isinstance(c, conditions.FunnelCondition):
                plotting.plot_funnel(self.ax, c.coef, c='tab:green',
                                     label='Trigger',lw=lw)
        for m in self.modification:
            if isinstance(m, modifications.FunnelModification):
                plotting.plot_funnel(self.ax, m.coef1, c='tab:orange',
                                     label='Reset',lw=lw)
        legend = self.ax.legend(frameon=False,
                                bbox_to_anchor=(0., 0., 0.5, 0.5),
                                ncol=2,
                                columnspacing=0.3,
                                handletextpad=0.3,
                                labelspacing=0.3,
                                borderaxespad=0.3)
        for text in legend.get_texts():
            text.set_color('k')
        if self.paper or self.presentation_video:
            hide_all_ticks(self.ax)
            plt.tight_layout(pad=0)

    def step(self, action):
        obs, r, done, info = super().step(action)
        # Plot if we are at skipframe or if teacher intervenes
        intervention = info['intervention']
        crash = self.env.game_over
        landed = not self.env.lander.awake
        timeout = self.env.num_steps >= self.env.timeout
        if np.mod(self.episode_steps, self.skipframe) == 0 or intervention or \
                crash or landed or timeout:
            self._plot()
            if not self.paper:
                if crash:
                    self.ax.set_facecolor('b')
                elif landed:
                    self.ax.set_facecolor('y')
                elif timeout:
                    self.ax.set_facecolor('c')
                elif intervention:
                    self.ax.set_facecolor('r')
                else:
                    self.ax.set_facecolor('w')
            if self.dirname is not None:
                if self.paper and not self.presentation_video:
                    plt.savefig(os.path.join(self.dirname,
                                             f'{self.n_figures}.pdf'),
                                format='pdf', transparent=True)
                else:
                    plt.savefig(
                        os.path.join(self.dirname, f'{self.n_figures}.png'))
            self.n_figures += 1

        self.episode_steps += 1

        return obs, r, done, info

    def reset(self, **kwargs):
        self.episode_steps = 0
        obs = super().reset(**kwargs)
        self.get_background()
        return obs

    def close(self):
        # Make video
        if self.dirname is not None:
            current_dir = os.getcwd()
            os.chdir(self.dirname)
            subprocess.call(
                ['ffmpeg', '-y', '-framerate', '32', '-i', '%d.png', '-r', '30',
                 '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
                 'video_name.mp4'])
            # Clean png files
            for file in os.listdir(self.dirname):
                if file.endswith('.png'):
                    os.remove(file)

            os.chdir(current_dir)
        super().close()


class MonitorVideoIntervention(CustomMonitorVideo):
    def __init__(self, env, dirname=None, skipframe=10, paper=False, presentation_video=False):
        """
        Parameters
        ----------
        env: LunarLander interventions
        dirname: str
            Name of the folder where to save the video
        skipframe: int
            We plot every skipframe frames for efficiency
        paper: bool
            Whether to produce the plots for the paper

        This class is different from the original monitor because it
        separates the interventions (called modified_env) from the
        underlying lunar lander environment (called env) to be able to reuse as
        much as possible the original env but making the right calles to
        step (for the constraint) and to reset (to restart the intervention
        timeout). By calling the underlying env, self.env, we can keep the
        same plotting function.
        """
        super().__init__(env, dirname, skipframe, paper, presentation_video)
        # We distinguish between the intervention wrapper (called for step)
        # from the basic lunar lander env (called for lander position and so
        # on)
        self.modified_env = self.env
        self.env = self.modified_env.env

    def step(self, action):
        obs, r, g, done, info = self.modified_env.step(action)
        # Plot if we are at skipframe or if teacher intervenes
        intervention = info['teacher_intervention']
        crash = self.env.game_over
        landed = not self.env.lander.awake
        timeout = self.modified_env.num_steps >= self.modified_env.timeout
        if np.mod(self.episode_steps, self.skipframe) == 0 or intervention or \
                crash or landed or timeout:
            self._plot()
            if not self.paper:
                if crash:
                    self.ax.set_facecolor('r')
                elif landed:
                    self.ax.set_facecolor('y')
                elif timeout:
                    self.ax.set_facecolor('c')
                elif intervention:
                    self.ax.set_facecolor('b')
                else:
                    self.ax.set_facecolor('w')
            if self.dirname is not None:
                if self.paper:
                    plt.savefig(os.path.join(self.dirname,
                                             f'{self.n_figures}.pdf'),
                                format='pdf', transparent=True)
                else:
                    plt.savefig(os.path.join(self.dirname, f'{self.n_figures}.png'),
                                dpi=400)
            self.n_figures += 1

        self.episode_steps += 1

        return obs, r, g, done, info

    def reset(self, **kwargs):
        self.episode_steps = 0
        # We cannot call super().reset() because it would call the reset
        # method of the original lander and not of the interventions since
        # we assigned self.evn to the original lander to reuse the other
        # monitor
        obs = self.modified_env.reset(**kwargs)
        self.get_background()
        return obs


W = VIEWPORT_W / SCALE
CHUNKS = 11
chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
helipad_x1 = chunk_x[CHUNKS // 2 - 1]
helipad_x2 = chunk_x[CHUNKS // 2 + 1]
l_heli, _ = absolute_coord_to_landing_coord(helipad_x1, 0)
r_heli, _ = absolute_coord_to_landing_coord(helipad_x2, 0)


def inside_helipad(x):
    return np.logical_and(x >= l_heli, x <= r_heli)


def y_envelop(x, coef):
    if x <= l_heli:
        return coef * (l_heli - x)
    elif x >= r_heli:
        return coef * (x - r_heli)
    else:
        return 0


def set_lander_pos(xland, yland, state):
    """
    Compute the full state such that the lander is in the specified xland,
    yland coordinate.

    The desired coordinates are specified in the reference frame centered at the
    helipad while the state is specified in the absolute reference frame of
    the plotting window. The reason why we need the full state is that we
    compute the new position of the legs by applying a shift to their
    original position so we do not have to worry about breaking the lander
    geometry.

    """
    xland_abs, yland_abs = landing_coord_to_absolute_coord(xland, yland)
    xshift = xland_abs - state[0, 0]
    yshift = yland_abs - state[0, 1]
    new_state = state.copy()
    new_state[:, 0] += xshift
    new_state[:, 1] += yshift
    return new_state
