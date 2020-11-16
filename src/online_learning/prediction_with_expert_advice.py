import numpy as np
import matplotlib.pyplot as plt
from src.online_learning import ExponetiatedGradient


def create_experts(n, mean_range, std_range):
    means = mean_range[0] + (mean_range[1] - mean_range[0]) * np.random.rand(n)
    stds = std_range[0] + (std_range[1] - std_range[0]) * np.random.rand(n)
    return lambda t: np.random.multivariate_normal(means, np.diag(stds) ** 2)


def sample_from_mixture(means, stds):
    n_components = len(means)
    i = np.random.randint(0, n_components, 1)[0]
    return means[i] + stds[i] * np.random.randn(1)


def main():
    means = [-1, 1, 6]
    stds = [0.1, 0.5, 2]

    n_rounds = 1000
    n_experts = 50
    experts = create_experts(n_experts, [-10, 10], [0.3, 3])

    ol = ExponetiatedGradient(n_experts, 1, 0.25)
    ol_loss = np.zeros(n_rounds)
    ol_ranking = np.zeros(n_rounds)

    loss = np.zeros((n_rounds, n_experts))

    for i in range(n_rounds):
        exp_advice = experts(i)
        reference = sample_from_mixture(means, stds)
        z = np.abs(exp_advice - reference)
        loss[i, :] = z

        ol_loss[i] = np.dot(ol.w, z)
        ol_ranking[i] = 1 + np.sum(np.sum(loss[:i+1, :], axis=0) <= np.sum(ol_loss[:i+1]))
        ol.step(z)

    plt.plot(np.cumsum(loss, axis=0), '--')
    plt.plot(np.cumsum(ol_loss), lw=2)
    plt.figure()
    plt.plot(ol_ranking)
    plt.show()


if __name__ == '__main__':
    main()