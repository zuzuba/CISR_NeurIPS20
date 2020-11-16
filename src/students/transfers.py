import numpy as np
import re
from stable_baselines.a2c.utils import ortho_init

__all__ = ['identity_transfer', 'reset_ppo_vf']


def identity_transfer(params):
    new_params = params.copy()
    return new_params


def init_weights(scale, shape):
    init_function = ortho_init(scale)
    return init_function(shape)


def init_bias(shape):
    return np.zeros(shape)


def reset_ppo_vf(params):
    new_params = params.copy()

    e = re.compile('model/vf')
    for name, val in new_params.items():
        if re.match(e, name):
            substr = re.sub(e, '', name)

            if re.match('.*/w', substr) is not None:
                val[:] = init_weights(np.sqrt(2), val.shape)[:]
            elif re.match('.*/b', substr) is not None:
                val[:] = init_bias(val.shape)[:]
            else:
                raise RuntimeWarning('Cannot initialize this typer of '
                                     'variable')
    return new_params
