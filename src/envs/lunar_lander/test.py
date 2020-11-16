import numpy as np
from numpy.testing import *
import unittest
from collections import deque

import src.envs.lunar_lander.modifications as modifications
import src.envs.lunar_lander.conditions as conditions
import src.envs.lunar_lander.utils as utils
import src.envs.lunar_lander.interventions as interventions

from gym.envs.box2d.lunar_lander import VIEWPORT_H, VIEWPORT_W, SCALE, \
    LEG_DOWN, FPS


class TestConditions(unittest.TestCase):
    def test_funnel_condition(self):
        # No coef
        c = conditions.FunnelCondition()
        assert not c.verify([0, 1, 0, 0, 0, 0, 0, 0])

        # With coef on the left
        offset = LEG_DOWN/SCALE/ (VIEWPORT_H/SCALE/2) # Offset cause y_heli is not at zero height
        c = conditions.FunnelCondition(1)
        assert not c.verify([0, 1, 0, 0, 0, 0, 0, 0]) # Inside helipad
        assert c.verify([-0.3, 0.095 + offset, 0, 0, 0, 0, 0, 0])  # Below funnel
        assert not c.verify([-0.3, 0.105 + offset, 0, 0, 0, 0, 0, 0]) # Above
        # funnel
        assert c.verify([-1.2, 1, 0, 0, 0, 0, 0, 0]) # Out of map

        # With coef on the right
        assert c.verify(
            [0.3, 0.095 + offset, 0, 0, 0, 0, 0, 0])  # Below funnel
        assert not c.verify([0.3, 0.105 + offset, 0, 0, 0, 0, 0, 0])  # Above
        # funnel
        assert c.verify([1.2, 1, 0, 0, 0, 0, 0, 0])  # Out of map

    def test_yvel_helipad_condition(self):
        c = conditions.YVelHelipadCondition()
        assert not c.verify([-0.3, 1, 0, 0, 0, 0, 0, 0])  # Outside helipad
        assert c.verify([0, 0.01, 0, -0.45, 0, 0, 0, 0])  # Too fast
        assert not c.verify([0, 0.01, 0, -0.35, 0, 0, 0, 0])  # Slow enough

    def test_minimal_condition(self):
        pass


class TestModifications(unittest.TestCase):
    def test_stabilize(self):
        m = modifications.Stabilize()
        s = np.ones((3, 6), dtype=np.float)
        news = m.get_new_state(s)
        assert_almost_equal(news, np.hstack((np.ones((3, 2)),
                                             np.zeros((3, 4)))))

    def test_identity(self):
        m = modifications.IdentityModification()
        s = np.ones((3, 6), dtype=np.float)
        news = m.get_new_state(s)
        assert_almost_equal(news, np.ones((3, 6)))

    def test_inside_heli_modification(self):
        m = modifications.YPosModificationInsideHeli(0.1)
        # Inside heli
        s = np.zeros((3, 6), dtype=np.float)
        x, y = utils.landing_coord_to_absolute_coord(0, 0)
        s[:, 0] = x
        s[:, 1] = y
        news = m.get_new_state(s)
        target_s = s.copy()
        _, y = utils.landing_coord_to_absolute_coord(0, 0.1)
        target_s[:, 1] = y
        assert_almost_equal(news, target_s)

        # Outside heli
        s = np.zeros((3, 6), dtype=np.float)
        x, y = utils.landing_coord_to_absolute_coord(-0.3, 0)
        s[:, 0] = x
        s[:, 1] = y
        news = m.get_new_state(s)
        assert_almost_equal(news, s)

    def test_funnel_modification(self):
        pass

    def test_get_xy_reset(self):
        coef1, coef2, xold, yold = 1, 1, -0.3, 1.7
        xnew, ynew = modifications.get_xy_reset(coef1, coef2, xold, yold)
        assert_almost_equal([xnew, ynew], [-1.1, 0.9])


class TestInterventions(unittest.TestCase):
    def test_backtracking_intervention(self):
        interv = interventions.LunarBacktrackingIntervention(
            history_length=2, cond_c=1)
        buffer = deque([np.zeros((3, 6)), np.ones((3, 6))])

        # Too fast
        s = np.zeros((3, 6))
        x, y = utils.landing_coord_to_absolute_coord(0, 0.1)
        s[:, [0, 1, 3]] = [x, y, -1.4 * 1 / ((VIEWPORT_H / SCALE / 2) / FPS)]
        interv.reset()
        interv.state_buffer = buffer.copy()
        interv.env.set_state(s)
        obs, r, g, done, info = interv.step(0)
        assert_array_equal(g, [0, 1])
        assert_array_equal(interv.env.get_state(),
                           np.hstack((np.ones((3, 2)),
                                      np.zeros((3, 4)))))

        # Slow enough
        s = np.zeros((3, 6))
        x, y = utils.landing_coord_to_absolute_coord(0, 0.1)
        s[:, [0, 1, 3]] = [x, y, -1. * 1 / ((VIEWPORT_H / SCALE / 2) / FPS)]
        interv.reset()
        interv.env.set_state(s)
        s, r, g, done, info = interv.step(0)
        assert_array_equal(g, [0, 0])

        # Too tilted
        s = np.zeros((3, 6))
        x, y = utils.landing_coord_to_absolute_coord(0, 0.01)
        s[:, [0, 1, 4]] = [x, y, 0.65]
        interv.reset()
        interv.state_buffer = buffer.copy()
        interv.env.set_state(s)
        obs, r, g, done, info = interv.step(0)
        assert_array_equal(g, [0, 1])
        assert_array_equal(interv.env.get_state(),
                           np.hstack((np.ones((3, 2)),
                                      np.zeros((3, 4)))))

        # Straight enough
        s = np.zeros((3, 6))
        x, y = utils.landing_coord_to_absolute_coord(0, 0.01)
        s[:, [0, 1, 4]] = [x, y, 0.45]
        interv.reset()
        interv.env.set_state(s)
        obs, r, g, done, info = interv.step(0)
        assert_array_equal(g, [0, 0])

        # Out of Funnel
        s = np.zeros((3, 6))
        x, y = utils.landing_coord_to_absolute_coord(-0.6, 0.35)
        s[:, [0, 1]] = [x, y]
        interv.reset()
        interv.state_buffer = buffer.copy()
        interv.env.set_state(s)
        obs, r, g, done, info = interv.step(0)
        assert_array_equal(g, [0, 1])
        assert_array_equal(interv.env.get_state(),
                           np.hstack((np.ones((3, 2)),
                                      np.zeros((3, 4)))))

    def test_orthogonal_intervention(self):
        interv = interventions.LanderOrthogonalIntervention(cond_c=1, mod_c1=3)

        # Too fast
        s = np.zeros((3, 6))
        x, y = utils.landing_coord_to_absolute_coord(0, 0.1)
        s[:, [0, 1, 3]] = [x, y, -1.4 * 1 / ((VIEWPORT_H / SCALE / 2) / FPS)]
        interv.reset()
        interv.env.set_state(s)
        obs, r, g, done, info = interv.step(0)
        assert_array_equal(g, [0, 0, 1])

        # Slow enough
        s = np.zeros((3, 6))
        x, y = utils.landing_coord_to_absolute_coord(0, 0.1)
        s[:, [0, 1, 3]] = [x, y, -1. * 1 / ((VIEWPORT_H / SCALE / 2) / FPS)]
        interv.reset()
        interv.env.set_state(s)
        obs, r, g, done, info = interv.step(0)
        assert_array_equal(g, [0, 0, 0])

        # Too tilted
        s = np.zeros((3, 6))
        x, y = utils.landing_coord_to_absolute_coord(0, 0.01)
        s[:, [0, 1, 4]] = [x, y, 0.65]
        interv.reset()
        interv.env.set_state(s)
        obs, r, g, done, info = interv.step(0)
        assert_array_equal(g, [0, 0, 1])
        _, new_y = utils.landing_coord_to_absolute_coord(0, .11)
        new_s = s.copy()
        new_s[:, [1, 4]] = [new_y, 0]
        # Use the tolerance because we are not comuting the y displacemente
        # downard before the intervention is applied. We compare only the
        # first row because get_state automatically shifts the legs
        assert_allclose(interv.env.get_state()[0, :], new_s[0, :], rtol=5e-3)

        # Straight enough
        s = np.zeros((3, 6))
        x, y = utils.landing_coord_to_absolute_coord(0, 0.01)
        s[:, [0, 1, 4]] = [x, y, 0.45]
        interv.reset()
        interv.env.set_state(s)
        obs, r, g, done, info = interv.step(0)
        assert_array_equal(g, [0, 0, 0])

        # Out of helipad, out of funnel
        s = np.zeros((3, 6))
        x, y = utils.landing_coord_to_absolute_coord(-0.8, 0.5)
        s[:, [0, 1]] = [x, y]
        interv.reset()
        interv.env.set_state(s)
        obs, r, g, done, info = interv.step(0)
        assert_array_equal(g, [0, 1, 0])
        x, y = utils.landing_coord_to_absolute_coord(
            *np.linalg.solve([[-1, 1], [3, 1]], [1.3, -0.6]))
        new_s = s.copy()
        new_s[:, [0, 1]] = [x, y]
        assert_allclose(interv.env.get_state()[0, :], new_s[0, :], rtol=5e-3)

        # Out of helipad, inside funnel
        offset = LEG_DOWN / SCALE / (VIEWPORT_H / SCALE / 2) # Offset cause y_heli is not at zero height
        s = np.zeros((3, 6))
        x, y = utils.landing_coord_to_absolute_coord(-0.6, 0.41 + offset)
        s[:, [0, 1]] = [x, y]
        interv.reset()
        interv.env.set_state(s)
        obs, r, g, done, info = interv.step(0)
        assert_array_equal(g, [0, 0, 0])

    def test_timeout(self):
        interv = interventions.LanderOrthogonalIntervention(
            cond_c=1, mod_c1=3,timeout=500)
        interv.num_steps = 499
        s, r, g, done, info = interv.step(0)
        assert r == -100
        assert done


class TestPerformanceInterventionsMonitor(unittest.TestCase):
    def setUp(self):
        def env_f(timeout):
            env = interventions.LunarOrthogonalInterventionMonitored(
                cond_c=1, mod_c1=3, timeout=timeout, filename=None)
            return env
        self.env_f = env_f

    def test_timouts(self):
        env = self.env_f(4)
        r_array = np.zeros(2)
        for i in range(2):
            env.reset()
            r = 0
            for _ in range(4):
                obs, r_tmp, g, done, info = env.step(0)
                r += r_tmp
            assert done
            r_array[i] = r
        assert_array_equal(env.get_episode_rewards(), r_array)
        assert_array_equal(env.get_episode_lengths(), [4, 4])
        assert_array_equal(env.get_episode_timeouts(), [True, True])
        assert_array_equal(env.get_episode_timeouts_on_ground(), [False, False])
        assert_array_equal(env.get_episode_oom(), [False, False])
        assert_array_equal(env.get_episode_crashes(), [False, False])
        assert_array_equal(env.get_episode_successes(), [False, False])
        assert_array_equal(env.get_episode_interventions(), [0, 0])

    def test_successes(self):
        env = self.env_f(80)
        r_array = np.zeros(2)
        for i in range(2):
            env.reset()
            # Set it at the origin (need -0.01 to avoid intervention
            # activating)
            new_state = utils.set_lander_pos(0, -0.01, env.env.get_state())
            new_state[:, 2:] = 0
            env.env.set_state(new_state)
            r = 0
            for _ in range(80):
                obs, r_tmp, g, done, info = env.step(0)
                r += r_tmp
                if done:
                    break
            assert done
            r_array[i] = r
        assert_array_equal(env.get_episode_rewards(), r_array)
        assert_array_equal(env.get_episode_timeouts(), [False, False])
        assert_array_equal(env.get_episode_timeouts_on_ground(),
                           [False, False])
        assert_array_equal(env.get_episode_oom(), [False, False])
        assert_array_equal(env.get_episode_crashes(), [False, False])
        assert_array_equal(env.get_episode_successes(), [True, True])
        assert_array_equal(env.get_episode_interventions(), [0, 0])

    def test_crashes_and_oom(self):
        env = interventions.LunarOrthogonalInterventionMonitored(
            cond_c=None, mod_c1=None, filename=None)
        # Crash
        env.reset()
        new_state = utils.set_lander_pos(0, -0.01, env.env.get_state())
        new_state[:, 3] = -2
        env.env.set_state(new_state)
        for i in range(10):
            _, _, _, done, _ = env.step(0)
            if done:
                break

        # OOM
        env.reset()
        new_state = utils.set_lander_pos(-0.95, 1, env.env.get_state())
        new_state[:, 2] = -3
        env.env.set_state(new_state)
        for i in range(10):
            _, _, _, done, _ = env.step(0)
            if done:
                break

        assert_array_equal(env.get_episode_timeouts(), [False, False])
        assert_array_equal(env.get_episode_timeouts_on_ground(),
                           [False, False])
        assert_array_equal(env.get_episode_oom(), [False, True])
        assert_array_equal(env.get_episode_crashes(), [True, False])
        assert_array_equal(env.get_episode_successes(), [False, False])
        assert_array_equal(env.get_episode_interventions(), [0, 0])

    def test_interventions(self):
        env = self.env_f(10)
        env.reset()
        new_state = utils.set_lander_pos(-0.4, 0.15, env.env.get_state())

        for i in range(10):
            env.env.set_state(new_state)
            _, _, _, done, _ = env.step(0)
            if done:
                break

        assert_array_equal(env.get_episode_timeouts(), [True])
        assert_array_equal(env.get_episode_timeouts_on_ground(),
                           [False])
        assert_array_equal(env.get_episode_oom(), [False])
        assert_array_equal(env.get_episode_crashes(), [False])
        assert_array_equal(env.get_episode_successes(), [False])
        assert_array_equal(env.get_episode_interventions(), [10])


if __name__ == '__main__':
    unittest.main()
