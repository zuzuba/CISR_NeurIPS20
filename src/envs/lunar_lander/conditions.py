import numpy as np
import src.envs.lunar_lander.utils as utils

from gym.envs.box2d.lunar_lander import VIEWPORT_H, SCALE, LEG_DOWN


__all__ = ['InterventionCondition',
           'FalseCondition',
           'MinimalCondition',
           'FunnelCondition',
           'YVelHelipadCondition']


class InterventionCondition(object):
    def __init__(self):
        pass

    def verify(self, obs):
        raise NotImplementedError

    def reset_env(self, env):
        pass


class FalseCondition(InterventionCondition):
    """
    Dummy condition for not intervening.
    """
    def verify(self, obs):
        return False


class MinimalCondition(InterventionCondition):
    """
    Minimal intervention to guarantee safety and allow landing.

    Inside the helipad x coordinates, it imposes a condition on angle and
    y-velocity. Outside the helipad, it creates a funnel of allowed x-y
    positions. The funnel is determined by the coef parameter and the height of
    the highest peak in the map.
    """
    def __init__(self, coef=None):
        # Height of highest mountain
        self.highest_y = 0
        self.coef = coef

    def y_envelop(self, x):
        if self.coef is None:
            return -1000
        else:
            return utils.y_envelop(x, self.coef)

    def verify(self, obs):
        out_of_helipad = not utils.inside_helipad(obs[0])
        ylim = np.maximum(self.highest_y, self.y_envelop(obs[0]))
        below_limit = obs[1] - LEG_DOWN/SCALE / (VIEWPORT_H/SCALE/2) <= ylim
        almost_oom = np.abs(obs[0]) >= 1 - 0.1
        condition1 = (out_of_helipad and below_limit) or almost_oom

        small_landing_vel = np.abs(obs[3]) < 0.3 + 10 * obs[1]
        small_landing_angle = np.abs(obs[4]) < .5 + 10 * obs[1]
        condition2 = (not out_of_helipad) and \
                     (not small_landing_vel or not small_landing_angle)
        return condition1 or condition2

    def reset_env(self, env):
        height_f = utils.get_height_function(env)
        x = np.linspace(-1, 1, 400)
        y = height_f(x)
        self.highest_y = np.max(y)


class FunnelCondition(InterventionCondition):
    """
    Minimal intervention to guarantee safety and allow landing.

    Inside the helipad x coordinates, it imposes a condition on angle and
    y-velocity. Outside the helipad, it creates a funnel of allowed x-y
    positions. The funnel is determined by the coef parameter and the height of
    the highest peak in the map.
    """
    def __init__(self, coef=None):
        self.coef = coef

    def y_envelop(self, x):
        if self.coef is None:
            return -1000
        else:
            return utils.y_envelop(x, self.coef)

    def verify(self, obs):
        out_of_helipad = not utils.inside_helipad(obs[0])

        ylim = self.y_envelop(obs[0])
        below_limit = obs[1] - LEG_DOWN/SCALE/ (VIEWPORT_H/SCALE/2) <= ylim
        almost_oom = np.abs(obs[0]) >= 1 - 0.1
        condition1 = (out_of_helipad and below_limit) or almost_oom
        return condition1


class YVelHelipadCondition(InterventionCondition):
    """
    Minimal intervention to guarantee safety and allow landing.

    Inside the helipad x coordinates, it imposes a condition on angle and
    y-velocity. Outside the helipad, it creates a funnel of allowed x-y
    positions. The funnel is determined by the coef parameter and the height of
    the highest peak in the map.
    """
    def __init__(self):
        pass

    def verify(self, obs):
        inside_of_helipad = utils.inside_helipad(obs[0])
        small_landing_vel = np.abs(obs[3]) < 0.3 + 10 * obs[1]
        small_landing_angle = np.abs(obs[4]) < .5 + 10 * obs[1]
        condition2 = inside_of_helipad and \
                     (not small_landing_vel or not small_landing_angle)
        return condition2
