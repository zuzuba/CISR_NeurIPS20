import numpy as np
import src.envs.lunar_lander.utils as utils

__all__ = ['Stabilize', 'IdentityModification', 'YPosModificationInsideHeli',
           'FunnelModification', 'get_xy_reset']


class Modification(object):
    def __init__(self):
        pass

    def get_new_state(self, state):
        raise NotImplementedError


class Stabilize(Modification):
    def __init__(self):
        """
        Set linear and angular velocities to zero as well as the angle.
        """

    def get_new_state(self, state):
        new_state = state.copy()
        new_state[:, [2, 3, 4, 5]] *= 0
        return new_state


class IdentityModification(Modification):
    def get_new_state(self, state):
        return state


class FunnelModification(Modification):
    def __init__(self, coef1, coef2):
        self.coef1 = coef1
        self.coef2 = coef2

    def get_new_state(self, state):
        if utils.inside_helipad(state[0, 0]):
            return state
        else:
            xland, yland = utils.absolute_coord_to_landing_coord(
                state[0, 0], state[0, 1])
            new_xland, new_yland = get_xy_reset(self.coef1, self.coef2, xland,
                                                yland)
            # The goal of this modification is to push the lander up and
            # toward the center. In case the modification is activated by
            # the highest mountain condition instead of the funner itself,
            # the geometric construction pushes in the opposite direction.
            # This hack prevents this from happening
            if np.abs(new_xland) > np.abs(xland):
                if xland > 0:
                    new_xland = xland - np.abs(new_xland - xland)
                else:
                    new_xland = xland + np.abs(new_xland - xland)
            if np.abs(new_yland) < np.abs(yland):
                new_yland = yland + np.abs(new_yland - yland)

            # Ensure there is a minimal displacement
            while np.linalg.norm([new_xland - xland, new_yland - yland]) < \
                    0.1:
                new_xland = xland + 2 * (new_xland - xland)
                new_yland = yland + 2 * (new_yland - yland)

            new_state = utils.set_lander_pos(new_xland, new_yland, state)
            new_state[:, [2, 3, 4, 5]] *= 0
        return new_state


def get_xy_reset(coef1, coef2, xold, yold):
    """
    Compute a new xy coord based on a geometric construction.

    The new xy coordinate is defined as the intersection of two lines. The
    first line is the landing funnel that has the inclination determined by
    coef1. The second line is that has an inclination such that it is
    orthogonal to coef2 and passes through xold, yold

    """
    # Funnel line with coef1
    # TODO: Understand why it breaks with xold < l_helipad and xold >  r_heli
    if xold < 0:
        b1 = coef1 * utils.l_heli
        m1 = - coef1
        m2 = 1 / coef2  # Coef of the perpendicular line
    elif xold > 0:
        b1 = -coef1 * utils.r_heli
        m1 = coef1
        m2 = -1 / coef2  # Coef of the perpendicular line

    # Offset of line with coef perpendicular to coef 2 passing through xold,
    # yold
    b2 = yold - m2 * xold

    xprime = (b2 - b1) / (m1 - m2)
    yprime = m1 * xprime + b1

    return xprime, yprime


class YPosModificationInsideHeli(Modification):
    def __init__(self, y_offset=0.1):
        self.y_offeset = y_offset

    def get_new_state(self, state):
        xland, yland = utils.absolute_coord_to_landing_coord(state[0, 0],
                                                             state[0, 1])
        if ~utils.inside_helipad(xland):
            return state
        else:
            new_yland = yland + self.y_offeset
            new_state = utils.set_lander_pos(xland, new_yland, state)
            new_state[:, [2, 3, 4, 5]] *= 0
        return new_state
