import numpy as np
from stable_baselines.common.save_util import (
    is_json_serializable, data_to_json, json_to_data, params_to_bytes, bytes_to_params
)
import os


class ExponetiatedGradient(object):
    def __init__(self, d, B=1, eta=.5):
        """
        Initialize the exponentiated gradient method.

        Parameters
        ----------
        d: int
            problem dimensionality
        B:
            Bound on the norm of the decision variable that determines the domain
        eta: float
            Learning rate
        """
        self.d = d
        self.B = B
        self.eta = eta
        self.w = np.full(d, B/d)

    def step(self, z):
        z = np.clip(z, -200, np.inf)  # Excessively small values make unnormalized weights go to infinity and create numerical instabilities
        unnormalized_w = self.w * np.exp(-self.eta * z)
        self.w = self.B * unnormalized_w / np.sum(unnormalized_w)
        if np.any(np.isnan(self.w)):
            raise RuntimeError('Weights contain Nan values')
        return self.w

    def get_parameters(self):
        return {"w": self.w}

    def load_parameters(self, params):
        self.w = params['w']