import numpy as np


class SingleSwitchPolicy(object):
    # TODO: add reset policy and do not pop from original list of actions but
    #  from a copy instead
    def __init__(self, thresholds, available_actions=None, name=None):
        self.thresholds = np.atleast_2d(thresholds)
        self.t_index = 0
        if available_actions is None:
            self.available_actions = list(range(self.thresholds.shape[0] + 1))
        else:
            self.available_actions = list(available_actions)
        assert len(self.available_actions) > self.thresholds.shape[0], \
            'Need more actions than thresholds'
        if name is not None:
            self.name = name
        else:
            self.name = f"{'_'.join(str(x) for x in self.thresholds.reshape(-1))}_" \
                        f"{''.join(str(x) for x in self.available_actions)}"

    def predict(self, obs):
        if len(self.available_actions) > 1 and \
               obs[0] >= self.thresholds[self.t_index, 0] and \
               obs[1] <= self.thresholds[self.t_index, 1]:
            self.available_actions.pop(0)
            self.t_index += 1
        return self.available_actions[0], None

    def save(self, save_path):
        if not save_path.endswith('.npz'):
            save_path += '.npz'
        np.savez(save_path, thresholds=self.thresholds,
                 available_actions=self.available_actions, name=self.name)

    # To edit and test
    @classmethod
    def load(cls, load_path):
        if not load_path.endswith('.npz'):
            load_path += '.npz'
        data = np.load(load_path)
        model = cls(thresholds=data['thresholds'],
                    available_actions=data['available_actions'],
                    name=str(data['name']))
        return model


def evaluate_single_switch_policy(policy, teacher_env, student_final_env,
                                  timesteps=int(1e4)):
    # Train the student
    done = False
    obs = teacher_env.reset()
    action_sequence = []
    while not done:
        action, _ = policy.predict(obs)
        action_sequence.append(action)
        obs, r, done, info = teacher_env.step(action)

    # Deploy in final env
    student = teacher_env.student
    obs = student_final_env.reset()
    rewards = 0
    rewards_ep = []
    for i in range(timesteps):
        action, _ = student.predict(obs, deterministic=False)
        obs, r, g, done, info = student_final_env.step(action)
        rewards += r
        if done:
            rewards_ep.append(rewards)
            rewards = 0
            obs = student_final_env.reset()

    rewards_ep = np.asarray(rewards_ep)
    return np.mean(rewards_ep)