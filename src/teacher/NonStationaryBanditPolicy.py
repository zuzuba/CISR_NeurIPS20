import numpy as np
from collections import deque


class NonStationaryBanditPolicy(object):
    """
    Class implementing policy from 'Teacher-student curriculum learning'
    Matiisen et al. 2019.
    """
    def __init__(self, n_tasks, K):
        self.buffer = [deque([], maxlen=K) for _ in range(n_tasks)]

    def predict(self, obs):
        np.random.seed()  # Otherwise always same policy with multiprocessing

        ind = np.nonzero([len(buf) == 0 for buf in self.buffer])[0]
        if len(ind) > 0:
            action = np.random.choice(ind)
        else:
            sample_r = [np.abs(np.random.choice(buf)) for buf in self.buffer]
            action = np.argmax(sample_r)

        return action, None
    
    def add_point(self, action, r):
        if not np.isnan(r):
            self.buffer[action].append(r)