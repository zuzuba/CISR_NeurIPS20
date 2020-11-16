import numpy as np
from gym.envs.box2d.lunar_lander import LunarLander

from gym.envs.box2d.lunar_lander import VIEWPORT_W, \
    VIEWPORT_H, SCALE, LEG_DOWN, FPS

from src.envs.CMDP import CMDP

__all__ = ['LunarLanderCustom',
           'LunarLanderCCustom',
           'LunarLanderCMDP',
           'LunarLanderCCMDP']


class LunarLanderCustom(LunarLander):
    """
    Lunar Lander environment that implements get_state, set_state,
    compute_obs and compute_reward.
    """

    def __init__(self, sensor_noise=None):
        if sensor_noise is not None:
            sensor_noise = np.asarray(sensor_noise)
            if sensor_noise.size == 6:
                sensor_noise = np.append(sensor_noise, [0, 0])
            sensor_noise[-2:] = 0  # No noise on the contact sensors
            assert sensor_noise.size == 8
        self.sensor_noise = sensor_noise
        self.noiseless_obs = np.zeros((8,), dtype=np.float32)
        super().__init__()

    def get_state(self):
        """
        Get position, velocity, angle, angular velocity of body and legs.
        """
        state = np.zeros((3, 6), dtype=float)
        for i, body in enumerate(self.drawlist):
            state[i, :] = [body.position.x,
                           body.position.y,
                           body.linearVelocity.x,
                           body.linearVelocity.y,
                           body.angle,
                           body.angularVelocity]
        return state

    def set_state(self, state):
        """
        Set position, velocity, angle, angular velocity of body and legs.
        """
        for body, s in zip(self.drawlist, state):
            body.position.x = s[0]
            body.position.y = s[1]
            body.linearVelocity.x = s[2]
            body.linearVelocity.y = s[3]
            body.angle = s[4]
            body.angularVelocity = s[5]

    def compute_obs(self):
        """
        Compute the observation given the current state (useful when setting state)
        """
        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (
                        VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
        ]

        state = np.array(state, dtype=np.float32)
        self.noiseless_obs[:] = state[:]

        if self.sensor_noise is not None:
            state += np.random.multivariate_normal(
                np.zeros_like(state), np.diag(self.sensor_noise))
        return state

    def compute_shaping(self):
        """
        Computes the reward shaping that encourages reaching the origin.
        Useful when resetting state.
        """
        state = self.compute_obs()
        reward = 0
        shaping = \
            - 100 * np.sqrt(state[0] * state[0] + state[1] * state[1]) \
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3]) \
            - 100 * abs(state[4]) + 10 * state[6] + 10 * state[
                7]  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # We remove this for now. This means that if the teacher intervenes,
        # it does not matter how much fuel we spent
        # reward -= m_power * 0.30  # less fuel spent is better, about -30 for heurisic landing
        # reward -= s_power * 0.03
        return reward

    def step(self, action):
        obs, r, done, info = super().step(action)
        obs = self.compute_obs()  # Introduce noise
        return obs, r, done, info


class LunarLanderCCustom(LunarLanderCustom):
    continuous = True


def lunar_lander_constraint(observation, env, **kwargs):
    crash = env.game_over
    oom = abs(observation[0]) >= 1.0
    return {'g': float(np.logical_or(crash, oom))}


class LunarLanderCMDP(CMDP):
    def __init__(self):
        env = LunarLanderCustom()
        super(LunarLanderCMDP, self).__init__(env, lunar_lander_constraint,
                                              0, 1, False)


class LunarLanderCCMDP(CMDP):
    def __init__(self):
        env = LunarLanderCCustom()
        super(LunarLanderCCMDP, self).__init__(env, lunar_lander_constraint,
                                               0, 1, False)


def main():
    env = LunarLanderCMDP()
    obs = env.reset()
    n_steps = 1000

    for i in range(n_steps):
        if np.any(obs[-2:] == 1):
            a = 0
        else:
            a = env.action_space.sample()
        obs, _, g, done, _ = env.step(a)
        # env.env.render()
        if done:
            if not env.lander.awake:
                print('Successful landing')
            if g > 0:
                print('Catastrophe!!')
            env.reset()


if __name__ == '__main__':
    main()

