from collections import Counter

import numpy as np
from matplotlib import pyplot as plt, patches as mpatches

from src.envs.CMDP import CMDP
from src.envs.frozen_lake.flake_constants import LEFT, DOWN, RIGHT, UP, \
    REWARD_MAPPING

__all__ = ['create_intervention_from_dist',
           'create_intervention_from_map',
           'OptimalAgent',
           'value_iteration',
           'actions_for_path',
           'one_step_reach',
           'add_teacher',
           'plot_map',
           'create_reward_shaping_map']


def create_intervention_from_dist(base_map, dist):
    intervention_map = add_teacher(base_map, n_steps=dist, danger='H',
                                   empty='F', teacher='T')
    return create_intervention_from_map(intervention_map)


def create_intervention_from_map(intervention_map):
    if isinstance(intervention_map, list):
        intervention_map = np.array([list(s) for s in intervention_map])

    def intervention_condition(env, **kwargs):
        row, col = np.unravel_index(env.get_state(), intervention_map.shape)
        return intervention_map[row][col] == 'T'

    return intervention_condition


class OptimalAgent(object):
    """
    Agent that follows a desired path or goes to goal is not path is specified.

    The agent has a policy that is optimal either to follow a specified path or
    to go to goal.

    Parameters
    ----------
    world_map: np.ndarray
        Map of the world with obstacles, start and goal
    path: np.ndarray
        Path that we want to follow, if None we take optimal path to goal
    not_slipping_prob: float
        Probability of going where intended
    reward_map: dict
        indicated the reward of empty cells, goal, danger
    epsilon: float
        value for epsilon greedy policy
    """
    def __init__(self, world_map, path=None, not_slipping_prob=1,
                 reward_map=None, epsilon=0):

        self.world_map = world_map
        if path is not None and path.ndim == 1:
            path = np.column_stack(np.unravel_index(path, world_map.shape))
        self.reward_map = REWARD_MAPPING if reward_map is None else reward_map
        self.path = path
        self.epsilon = epsilon
        self.not_slipping_prob = not_slipping_prob

        self.policy, V = self.compute_policy()

        # Since we use a custom defined reward to find the policy following
        # a specific path, we need to recompute the value function with the
        # actual reward.
        if self.path is None:
            self.V = V
        else:
            r = np.full_like(self.world_map, 0, dtype=float)
            for k, v in self.reward_map.items():
                r[world_map==k] = v
            terminal_state_mask = np.logical_or(self.world_map==b'G',
                                                self.world_map == b'H')
            _, self.V = value_iteration(r, terminal_state_mask,
                                     self.not_slipping_prob,
                                     initial_policy=self.policy)

    def compute_policy(self):
        # Define reward to use in value iteration
        r = np.full_like(self.world_map, self.reward_map[b'F'], dtype=float)
        r[self.world_map == b'H'] = self.reward_map[b'H']
        r[self.world_map == b'G'] = self.reward_map[b'G']

        terminal_state_mask = np.logical_or(self.world_map==b'G',
                                            self.world_map == b'H')
        if self.path is None:
            actions, V = value_iteration(r, terminal_state_mask,
                                         not_slipping_prob=self.not_slipping_prob)
        else:
            r[self.path[:, 0], self.path[:, 1]] = self.reward_map[b'G']
            actions, V = value_iteration(r, terminal_state_mask,
                                         not_slipping_prob=self.not_slipping_prob)
            # Add actions to follow the specified path
            actions_path = actions_for_path(self.path)
            actions[self.path[:-1, 0], self.path[:-1, 1]] = actions_path
        return actions, V

    def step(self, obs):
        if np.random.rand(1) > self.epsilon:
            # Step with map observation
            if isinstance(obs, np.ndarray):
                x, y = [int(v) for v in np.nonzero(np.squeeze(obs) == 1.)]
            # Step with coordinate observation
            else:
                x, y = np.unravel_index(obs, self.world_map.shape)
            return self.policy[x, y]
        else:
            return np.random.choice(4)


def value_iteration(r,terminal_states_mask, not_slipping_prob=1,
                    initial_policy=None,
                    eps=0.1):
    """
    Value iteration for a deterministic map with a given reward function.

    Parameters
    ----------
    r: np.ndarray
        The reward corresponding to each state in the map
    terminal_states_mask: np.ndarray
        Array of boold indicating which locations in the map correspond to
        terminal states
    not_slipping_prob: float
        Probability of going where intended
    initial_policy: np.ndarray
        policy that we want to compute the value for. If None, we compute
        the value of the optimal policy
    eps: float
        Tolerance to break the value iteration procedure.

    Returns
    -------
    actions: np.ndarray
        Array with same shape as r containing the actions to take in each state
    V: np.np.ndarray
        Array containing the value of each state
    """
    V = np.copy(r).astype(np.float)
    for _ in range(200):
        Vdet_u = np.copy(V)
        Vdet_r = np.copy(V)
        Vdet_l = np.copy(V)
        Vdet_d = np.copy(V)

        Vdet_u[1:, :] = V[:-1, :]
        Vdet_r[:, :-1] = V[:, 1:]
        Vdet_l[:, 1:] = V[:, :-1]
        Vdet_d[:-1, :] = V[1:, :]

        nsp = not_slipping_prob
        Vnext = np.zeros((4, )+r.shape, dtype=float)
        Vnext[UP, :, :] = nsp * Vdet_u + 0.5 * (1-nsp) * Vdet_l + 0.5 * (
                1-nsp) * Vdet_r
        Vnext[DOWN, :, :] = nsp * Vdet_d + 0.5 * (1-nsp) * Vdet_l + 0.5 * (
                1-nsp) * Vdet_r
        Vnext[LEFT, :, :] = nsp * Vdet_l + 0.5 * (1-nsp) * Vdet_u + 0.5 * (
                1-nsp) * Vdet_d
        Vnext[RIGHT, :, :] = nsp * Vdet_r + 0.5 * (1-nsp) * Vdet_u + 0.5 * (
                1-nsp) * Vdet_d
        if initial_policy is None:
            policy = np.argmax(r[None, :, :] + Vnext, axis=0)
        else:
            policy = np.copy(initial_policy)

        Vnew = np.copy(V)
        for a in [UP, DOWN, LEFT, RIGHT]:
            ind = np.logical_and(policy == a, ~ terminal_states_mask)
            Vnew[ind] = r[ind] + Vnext[a][ind]
        if np.linalg.norm(V - Vnew, ord=np.inf) < eps:
            break
        V[:, :] = Vnew
    return policy, V


def actions_for_path(path):
    """
    Computes the actions necessary to follow a given path.
    Parameters
    ----------
    path: np.ndarray
        n_states x 2 array containing x, y coordinates of the path.
    """
    action_mapping = {(1, 0): DOWN, (-1, 0): UP, (0, -1): LEFT,
                      (0, 1): RIGHT}
    actions = np.zeros(path.shape[0] - 1)
    for i in range(actions.size):
        delta = tuple(path[i + 1, :] - path[i, :])
        actions[i] = action_mapping.get(delta)
        if actions[i] is None:
            raise ValueError('Non valid path')
    return actions


def one_step_reach(base_mask):
    """
    One step reachability operator for grid world with four actions.

    Given a mask of starting states
    [[False, False, False],
     [False, True,  False],
     [False, False, False]]

    rerturns the mask of one step reachable states
    [[False, True,  False],
     [True,  False, True ],
     [False, True,  False]]
    """
    n, m = base_mask.shape

    one_step_mask = np.zeros_like(base_mask, dtype=bool)

    # Move right and left
    one_step_mask |= np.hstack((np.zeros((n, 1), dtype=bool), base_mask[:, :-1]))
    one_step_mask |= np.hstack((base_mask[:, 1:], np.zeros((n, 1), dtype=bool)))
    # Move up and down
    one_step_mask |= np.vstack((np.zeros((1, m), dtype=bool), base_mask[:-1, :]))
    one_step_mask |= np.vstack((base_mask[1:, :], np.zeros((1, m), dtype=bool)))

    return one_step_mask


def add_teacher(base_map, n_steps=1, danger='H', empty='F', teacher='T'):
    """
    Add teacher to the base map such that it rescues within n_steps from danger.

    Parameters
    ----------
    base_map: list of strings
        map
    n_steps: int
        distance from danger when teacher should rescue
    danger: char
        symbol for danger
    empty: char
        symbol for empty space
    teacher: char
        Symbol for teacher

    Returns
    -------
    new_map: list of strings
        modified map where the emtpy spots at a given distance from danger have been replaced with the teacher.

    """
    if n_steps == 0:
        return base_map
    base_map = np.array([list(s) for s in base_map])

    base_danger_mask = base_map == danger
    base_empty_mask = base_map == empty

    previous_mask = np.copy(base_danger_mask)
    for _ in range(n_steps):
        reach_mask = one_step_reach(previous_mask)
        previous_mask[:] |= reach_mask[:]

    new_map = np.copy(base_map)
    new_map[reach_mask & base_empty_mask] = teacher
    new_map_list = [''.join(row) for row in new_map]
    return new_map_list


def plot_map(base_map, legend=False):
    base_map = np.array([list(s) for s in base_map])
    elements = np.unique(base_map)

    map_to_plot = np.zeros_like(base_map, dtype=float)

    for i, el in enumerate(elements):
        map_to_plot[base_map == el] = float(i)
    plt.tick_params(axis='both',
                   which='both',
                   bottom=False,
                   left=False,
                    labelbottom=False,
                    labelleft=False)
    im = plt.imshow(map_to_plot)
    if legend:
        values = np.unique(map_to_plot.ravel())
        colors = [im.cmap(im.norm(value)) for value in values]
        labels = ['Safe', 'Goal', 'Danger', 'Start', 'Teacher']
        patches = [
            mpatches.Patch(color=colors[i], label=labels[i])
            for i in range(len(values))]
        plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2,
                   borderaxespad=0., frameon=False)

    plt.clim(0, 4)


def create_reward_shaping_map(base_map, goal='G', empty='F', normalize=True):
    """
    Create a map containing rewards based on the closest goal cell.

    The returned array has the same shape as value but conains 0 for non-empty cells and (possibly) normalized Manhattan
    distances from the closest goal for empty cells.

    Parameters
    ----------
    base_map: list of str
        Initial map
    goal: char
        symbol denoting the goal
    empty: char or None
        symbol denoting empty spaces. If None, the distance is copmuted for every tile and not only empty spaces
    normalize: bool
        whether the reward values should be normalized to be between 0 and 1

    """
    # Get map
    base_map = np.array([list(s) for s in base_map])
    if type(base_map[0][0]) == bytes or type(base_map[0][0]) == np.bytes_:
        goal = bytes(goal, 'utf-8')
        if empty is not None:
            empty = bytes(empty, 'utf-8')

    # Get locations
    goal_locations = np.column_stack(np.nonzero(base_map==goal)).reshape(1, -1, 2)
    if empty is not None:
        empty_locations = np.column_stack(np.nonzero(base_map == empty)).reshape(-1, 1, 2)
    else:
        empty_locations = np.column_stack(np.nonzero(np.ones_like(base_map))).reshape(-1, 1, 2)

    # Find minimal distance
    d = np.min(np.linalg.norm(goal_locations - empty_locations, ord=1, axis=2), axis=1)
    if normalize:
        d = d.astype(np.float)
        d /= np.max(d)
    dist_map = np.zeros_like(base_map, dtype=float)
    empty_locations = np.squeeze(empty_locations)
    dist_map[empty_locations[:, 0], empty_locations[:, 1]] = d
    return dist_map


def plot_trajectories(traj, world_shape):
    """
    Plot trajectories and state visitation frequency.

    Parameters
    ----------
    traj: list of ints (output of deploy function)
        list of trajectories
    world_shape: tuple

    """
    plt.figure(figsize=(20, 10))
    intensity_map = np.zeros(world_shape, dtype=int)
    occupancy = Counter(np.hstack(traj))
    for k, v in occupancy.items():
        intensity_map[np.unravel_index(k, world_shape)] = v
    plt.imshow(intensity_map)
    for t in traj:
        plt.plot(*np.unravel_index(t, world_shape)[::-1])


def deploy(model, env, timesteps=1000):
    """
    Deploy an agent in a frozen lake environemnt (MDP or CMDP) and measures
    performance.

    The performance is measured in terms of success rate, average return and
    average return conditioned on success.

    Parameters
    ----------
    model: stable_baselines model
    env: gym.env
        frozen lake environment
    timesteps: int

    Returns
    -------
    success_ratio: float
    avg_return: float
    avg_return_success: float
    trajectories: list of ints
        list of trajectories visited during deployment
    """

    obs = env.reset()
    reward_sum, length, successes, n_episodes = (0.0, 0, 0, 0)
    returns, returns_success, trajectories, trajectory = ([], [], [], [])

    for _ in range(timesteps):
        action, _ = model.predict(obs, deterministic=False)
        if isinstance(env, CMDP):
            obs, reward, g, done, info = env.step(action)
        else:
            obs, reward, done, info = env.step(action)
        reward_sum += reward
        length += 1
        trajectory.append(env.s)
        if done:
            success = info['next_state_type'] == 'G'
            successes += float(success)
            returns.append(reward_sum)
            if success:
                returns_success.append(reward_sum)
            length = 0
            reward_sum = 0.0
            n_episodes += 1
            obs = env.reset()
            trajectories.append(trajectory)
            trajectory = []
    if trajectory:
        trajectories.append(trajectory)
    if n_episodes == 0:
        n_episodes = 1
        returns.append(reward_sum)
    success_ratio = successes / n_episodes
    avg_return = np.mean(returns)
    avg_return_success = np.mean(returns_success)
    return success_ratio, avg_return, avg_return_success, trajectories


def plot_networks(student, shape):
    """
    Plot the student's policy and value for the flake environment with
    single state observation.

    Notice that this only works with the single state observation
    environment and not the whole map due to how the input to the networks
    are generated.

    Parameters
    ----------
    student: stable_baselines.PPO2
    shape: tuple
        Shape of the map
    """
    plt.figure()
    s = np.arange(np.prod(shape))
    plt.figure()
    value = student.train_model.value(s).reshape(shape)
    plt.imshow(value)

    pi = student.train_model.proba_step(s).T.reshape((-1,) + shape)
    x, y = np.unravel_index(s, shape)

    for a in range(pi.shape[0]):
        if a == UP:
            u = np.zeros_like(s)
            v = pi[a].T.ravel()
        if a == DOWN:
            u = np.zeros_like(s)
            v = -pi[a].T.ravel()
        if a == RIGHT:
            v = np.zeros_like(s)
            u = pi[a].T.ravel()
        if a == LEFT:
            v = np.zeros_like(s)
            u = -pi[a].T.ravel()
        plt.quiver(x, y, u, v)