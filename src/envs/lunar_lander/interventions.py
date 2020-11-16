import numpy as np
from collections import deque

from src.envs import CMDP
import src.envs.lunar_lander.conditions as conditions
import src.envs.lunar_lander.modifications as modifications
import src.envs.lunar_lander.utils as utils


from src.envs.lunar_lander.lunar_lander_custom import LunarLanderCustom, \
    LunarLanderCCustom, lunar_lander_constraint


class LanderIntervention(CMDP):
    def __init__(self, history_length=0, condition=None,
                 modification=None, taus=None, timeout=np.inf,
                 continuous=False, base_env_kwargs={}, **kwargs):
        """
        """

        if continuous:
            base_env = LunarLanderCCustom(**base_env_kwargs)
        else:
            base_env = LunarLanderCustom(**base_env_kwargs)


        # Define augmented constraints based on the teacher intervention
        if condition is None:
            self.condition = [conditions.FalseCondition()]
        else:
            assert isinstance(condition, list), 'A list of conditions is ' \
                                                'necessary'
            self.condition = condition

        def constraints(**constraint_kwargs):
            # Original constraint
            g = [lunar_lander_constraint(**constraint_kwargs)['g']]

            # Teacher constraint
            # obs = constraint_kwargs['observation']
            obs = constraint_kwargs['env'].noiseless_obs
            teacher_g = [float(cond.verify(obs)) for cond in self.condition]

            return_dict = dict(g=g+teacher_g)

            return return_dict

        # Add thresholds
        taus = taus if taus is not None else [0.] * len(self.condition)

        if not isinstance(taus, list):
            taus = list(taus)

        constraints_values = [0] + taus
        n_constraints = 1 + len(self.condition)

        # Init CMDP with original and teacher constraints
        super().__init__(base_env, constraints,
                         constraints_values,
                         n_constraints, avg_constraint=False)

        # Reset distributions
        self.history_length = history_length
        self.buf_size = self.history_length * 3
        if history_length < 0:
            self.state_buffer = None
        else:
            self.state_buffer = deque([], maxlen=self.buf_size)

        if modification is None:
            self.modification = [modifications.IdentityModification()]
        else:
            assert isinstance(modification, list), 'A list of modifications ' \
                                                   'is necessary'
            self.modification = modification

        assert len(self.condition) == len(self.modification),  \
            'A list of modifications with as many entries as conditions is ' \
            'necessary'

        self.timeout = timeout
        self.num_steps = 0

    def step(self, action):
        # Take a normal step in the env
        obs, reward, g, done, info = super().step(action)

        # We detect whether the teacher has intervened based on the
        # value of the constraint (assuming the conditions can only
        # return values that are > 0)
        teacher_constraint_array = g[1:]
        intervene = False
        for condition_index, v in enumerate(teacher_constraint_array):
            intervene |= v > 0
            if intervene:
                break
         # Reset from teacher
        if intervene:
            if self.state_buffer is not None:
                try:
                    # Pop all the states that you want to rewind
                    for _ in range(self.history_length - 1):
                        self.state_buffer.popleft()
                    # This state you were at history_length steps ago
                    state = self.state_buffer[0]
                # Call it done if there is no more time steps to rewind
                except IndexError:
                    reward = - 100
                    state = self.env.get_state()
                    done = True
            else:
                state = self.env.get_state()

            # Modify the state and add penalty
            new_state = self.modification[condition_index].get_new_state(state)
            self.env.set_state(new_state)
            obs = self.env.compute_obs()
            reward = self.env.compute_shaping()
        else:
            if self.state_buffer is not None:
                self.state_buffer.appendleft(self.env.get_state())

        self.num_steps += 1
        if self.num_steps >= self.timeout:
            done = True
            reward = -100

        info.update({'teacher_intervention': intervene})
        return obs, reward, g, done, info

    def reset(self, **kwargs):
        # Reset the buffers when the episode restarts
        if self.state_buffer is not None:
            self.state_buffer.clear()
        self.num_steps = 0
        obs = super().reset(**kwargs)
        if self.state_buffer is not None:
            self.state_buffer.appendleft(self.env.get_state())
        return obs


class LanderOrthogonalIntervention(LanderIntervention):
    def __init__(self, cond_c, mod_c1, taus=None, mod_c2=1, timeout=500,
                 continuous=False, base_env_kwargs={}, **kwargs):
        if cond_c is not None:
            condition = [conditions.FunnelCondition(cond_c),
                         conditions.YVelHelipadCondition()]
        else:
            condition = None
        if mod_c1 is not None:
            modification = [
                modifications.FunnelModification(mod_c1, mod_c2),
                modifications.YPosModificationInsideHeli()]
        else:
            modification = None

        super().__init__(history_length=-1,
                         condition=condition,
                         modification=modification,
                         taus=taus,
                         timeout=timeout,
                         continuous=continuous,
                         base_env_kwargs=base_env_kwargs,
                         **kwargs)


class LunarOrthogonalInterventionMonitored(
      utils.MonitorPerformanceIntervention):
    def __init__(self, cond_c, mod_c1, taus=None, mod_c2=1,
                 timeout=500, continuous=False, filename=None,
                 base_env_kwargs={}, **kwargs):
        env = LanderOrthogonalIntervention(cond_c=cond_c,
                                           mod_c1=mod_c1,
                                           taus=taus,
                                           mod_c2=mod_c2,
                                           timeout=timeout,
                                           continuous=continuous,
                                           base_env_kwargs=base_env_kwargs,
                                           **kwargs)
        super().__init__(env,
                         filename=filename,
                         allow_early_resets=True,
                         reset_keywords=(),
                         info_keywords=())


class LunarBacktrackingIntervention(LanderIntervention):
    def __init__(self, history_length, cond_c, taus=None, timeout=500,
                 continuous=False, base_env_kwargs={}, **kwargs):
        condition = [conditions.MinimalCondition(cond_c)]
        modification = [modifications.Stabilize()]
        super().__init__(history_length=history_length,
                         condition=condition,
                         modification=modification,
                         taus=taus,
                         timeout=timeout,
                         continuous=continuous,
                         base_env_kwargs=base_env_kwargs,
                         **kwargs)


class LunarBacktrackingInterventionMonitored(
      utils.MonitorPerformanceIntervention):
    def __init__(self, history_length, cond_c, taus=None, timeout=500,
                 continuous=False, filename=None, base_env_kwargs={},
                 **kwargs):
        env = LunarBacktrackingIntervention(history_length=history_length,
                                            cond_c=cond_c,
                                            taus=taus,
                                            timeout=timeout,
                                            continuous=continuous,
                                            base_env_kwargs=base_env_kwargs,
                                            **kwargs)
        super().__init__(env,
                         filename=filename,
                         allow_early_resets=True,
                         reset_keywords=(),
                         info_keywords=())


if __name__ == '__main__':
    """
    Example to use the intervention video monitor
    """
    import src.envs.lunar_lander.utils as utils
    import os
    # Saving location
    dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           os.pardir, os.pardir, os.pardir, 'videos',
                           'lunar_lander', 'narrow_funnel')
    os.makedirs(dirname, exist_ok=True)

    # Environment and monitor
    env = LanderOrthogonalIntervention(
        20, 100, timeout=200, base_env_kwargs={'sensor_noise': [0] * 8})
    env = utils.MonitorVideoIntervention(env, dirname=dirname, skipframe=10,
                                         paper=True)

    # Running
    obs = env.reset()
    for _ in range(600):
        obs, r, g, done, info = env.step(0)
        if done:
            env.reset()
    env.close()
