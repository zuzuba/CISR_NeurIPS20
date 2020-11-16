import numpy as np


__all__ = ['BaseEvaluationLogger', 'FrozenLakeEvaluationLogger']


class BaseEvaluationLogger(object):
    def __init__(self, n_interventions, n_constraint_per_interventions, record_trajectorires=True):
        # TODO: Default should be false for record trajectories and i should implement a set_record so that I can record them only when I am plotting and not when I am learning
        self.n_i = n_interventions
        self.n_c = n_constraint_per_interventions

        self.constraint_values = None
        self.rewards = None
        self.lagrangian_rewards = None
        self.episode_length = None
        self.termination = None
        self.trajectories = None

        self.constraint_values_episodes = None
        self.rewards_episodes = None
        self.lagrangian_rewards_episodes = None
        self.episode_lengths = None
        self.termination_episodes = None
        self.trajectories_episodes = None

        self.record_trajectories = record_trajectorires

        self.reset_counters()

    def reset_within_episode_counters(self):
        self.constraint_values = np.zeros((self.n_i, self.n_c), dtype=float)
        self.rewards = 0
        self.lagrangian_rewards = 0
        self.episode_length = 0
        self.termination = None
        if self.record_trajectories:
            self.trajectories = []

    def reset_across_episode_counters(self):
        self.constraint_values_episodes = []
        self.rewards_episodes = []
        self.lagrangian_rewards_episodes = []
        self.episode_lengths = []
        self.termination_episodes = []
        if self.record_trajectories:
            self.trajectories_episodes = []

    def reset_counters(self):
        self.reset_within_episode_counters()
        self.reset_across_episode_counters()

    @staticmethod
    def determine_termination_cause(transition_dict):
        if not transition_dict['done']:
            return None
        else:
            cause = -1 if transition_dict['info']['teacher_intervention'] else 0
            return cause

    def add_transition(self, transition_dict, lagrange_multipliers, g, constraint_values):
        self.constraint_values += constraint_values[0, :, :]
        r = transition_dict['reward']
        self.lagrangian_rewards += r - np.dot(lagrange_multipliers, g)
        self.rewards += r
        self.episode_length += 1
        self.termination = self.determine_termination_cause(transition_dict)

        if self.record_trajectories:
            self.trajectories.append(transition_dict['observation'])

        done = transition_dict['done']
        if done:
            self.update_across_episode_counters()

    def update_across_episode_counters(self):
        self.constraint_values_episodes.append(self.constraint_values)
        self.rewards_episodes.append(self.rewards)
        self.lagrangian_rewards_episodes.append(self.lagrangian_rewards)
        self.episode_lengths.append(self.episode_length)
        self.termination_episodes.append(self.termination)
        if self.record_trajectories:
            self.trajectories_episodes.append(self.trajectories)
        self.reset_within_episode_counters()

    def end_evaluation(self, last_transition, waste_episode=True):
        # Add the data from the last episode even if it is not complete
        done = last_transition['done']
        if not done:
            if waste_episode:
                if len(self.rewards_episodes) > 0:
                    pass
                else:
                    self.update_across_episode_counters()
            else:
                self.update_across_episode_counters()

    def get_counters(self):
        counters = {'constraint_values': np.asarray(self.constraint_values_episodes),
                    'rewards': np.asarray(self.rewards_episodes),
                    'lagrangian_rewards': np.asarray(self.lagrangian_rewards_episodes),
                    'lengths': np.asarray(self.episode_lengths),
                    'terminations': np.asarray(self.termination_episodes),
                    'trajectories': self.trajectories_episodes}
        return counters


class FrozenLakeEvaluationLogger(BaseEvaluationLogger):
    @staticmethod
    def determine_termination_cause(transition_dict):
        """Return -1 for failure, +1 for success and 0 for timeout"""
        if not transition_dict['done']:
            return None
        else:
            if transition_dict['info']['next_state_type'] == 'G':
                return 1
            elif transition_dict['info']['teacher_intervention']:
                return -1
            else:
                return 0
