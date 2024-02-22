import torch
import random
import matplotlib.pyplot as plt
from scipy.stats import norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


# Constantes
# Rangos para inicializar mu
MU_START = 10
MU_END = 50
# Rangos para inicializar sigma
SIGMA_START = 3.1
SIGMA_END = 6.2
# Dato para solucion heuristica
HEURISTIC_STEP = 5


def plot_observation(observation, show=False):
    mu = torch.mean(observation)
    sigma = torch.std(observation, unbiased=True)
    x_axis = torch.arange(min(observation) - 5, max(observation) + 5, 0.01)
    plt.scatter(observation.numpy(), torch.zeros(len(observation)), s=5, alpha=0.5)
    plt.plot(x_axis.numpy(), norm.pdf(x_axis.numpy(), mu.numpy(), sigma.numpy()),
             label=r'$\mu=' + str(round(mu.item(), 2)) + r',\ \sigma=' + str(round(sigma.item(), 2)) + r'$')
    if show:
        plt.legend()
        plt.show()


def generate_data(n_observations: int, k_parameters=2, show=False, heuristic=False):
    gaussian_distributions = []
    heuristic_mu = random.uniform(MU_START, MU_END) if heuristic else 0
    for k in range(k_parameters):
        mu = torch.tensor(random.uniform(MU_START, MU_END)) if not heuristic else torch.tensor(heuristic_mu +
                                                                                               HEURISTIC_STEP)
        heuristic_mu += HEURISTIC_STEP if heuristic else 0
        sigma = torch.tensor(random.uniform(SIGMA_START, SIGMA_END))
        normal_dist = torch.distributions.Normal(mu, sigma)
        sample = normal_dist.sample((n_observations, 1)).squeeze()
        gaussian_distributions.append(sample)
    for distribution in gaussian_distributions:
        plot_observation(distribution)
    if show:
        plt.legend()
        plt.show()
    return gaussian_distributions


MU_SPREAD_COEFFICIENT = 20
MU_SHIFT_COEFFICIENT = 30

SIGMA_SPREAD_COEFFICIENT = 1.5
SIGMA_SHIFT_COEFFICIENT = 3


def generate_data_improved(n_observations: int, k_parameters: int = 2):
    mus = torch.randn(k_parameters) * MU_SPREAD_COEFFICIENT + MU_SHIFT_COEFFICIENT
    sigmas = torch.abs(torch.randn(k_parameters) * SIGMA_SPREAD_COEFFICIENT + SIGMA_SHIFT_COEFFICIENT)
    # Create distributions
    distributions = torch.distributions.Normal(mus, sigmas)
    # Exact samples
    samples = distributions.sample(torch.Size([n_observations,])).t()
    return samples


generate_data_improved(20, 3)
