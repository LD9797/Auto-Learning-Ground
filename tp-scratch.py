import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gamma
from enum import Enum

MU_SPREAD_COEFFICIENT = 20
MU_SHIFT_COEFFICIENT = 30
SIGMA_SPREAD_COEFFICIENT = 1.5
SIGMA_SHIFT_COEFFICIENT = 3

pallet = ["#1F77B4", "#B41F77", "#77B41F"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


class Distributions(Enum):
    GAUSSIAN = "GAUSSIAN"
    GAMMA = "GAMMA"


def plot_observation(observation, show=True, color=pallet[0], obs_number=1, obs_type="Gaussian"):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    mu = torch.mean(observation)
    sigma = torch.std(observation, unbiased=True)
    x_axis = torch.arange(observation.min(), observation.max(), 0.01)
    ax1.scatter(observation, torch.zeros(observation.size()), s=3, alpha=0.5, color=color)
    label = fr'$\mu={round(mu.item(), 2)},\ \sigma={round(sigma.item(), 2)}$'
    ax1.plot(x_axis, norm.pdf(x_axis, mu, sigma), color=color, label=label)
    ax1.legend()
    ax2.hist(observation, bins=20, alpha=0.5, color=color)
    ax1.set_title(f"{obs_type} generated Observation #{obs_number}")
    if show:
        plt.show()


def plot_observation_2(observation, show=True, color=pallet[0], obs_number=1, obs_type=Distributions.GAUSSIAN):
    fig, ax = plt.subplots()
    x_axis = torch.arange(observation.min(), observation.max(), 0.01)
    # Histogram
    ax.hist(observation, density=True, bins=20, alpha=0.5, color=color)
    # Scatter Dots
    ax.scatter(observation, torch.zeros(observation.size()), s=6, alpha=0.5, color=color)
    # Gaussian Curve
    if obs_type == Distributions.GAUSSIAN:
        mu = torch.mean(observation)
        sigma = torch.std(observation, unbiased=True)
        label = fr'$\mu={round(mu.item(), 2)},\ \sigma={round(sigma.item(), 2)}$'
        ax.plot(x_axis, norm.pdf(x_axis, mu, sigma), color=color, label=label)
    elif obs_type == Distributions.GAMMA:
        # Estimate the parameters of the Gamma distribution
        alpha_hat, loc_hat, beta_hat_inv = gamma.fit(observation)
        beta_hat = 1 / beta_hat_inv  # Convert rate to scale
        # Print estimated parameters
        print(f"Estimated alpha (shape): {alpha_hat}")
        print(f"Estimated beta (scale): {beta_hat}")
        label = ""
        ax.plot(x_axis, gamma.pdf(x_axis, a=alpha_hat, scale=beta_hat), color=color, label=label)
    ax.set_title(f"{obs_type} generated Observation #{obs_number}")
    ax.legend()
    # Adjust y-axis
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - 0.01, y_max)
    if show:
        plt.show()


def generate_data_gaussian(n_observations: int, k_parameters: int = 2) -> torch.Tensor:
    mus = torch.randn(k_parameters) * MU_SPREAD_COEFFICIENT + MU_SHIFT_COEFFICIENT
    sigmas = torch.abs(torch.randn(k_parameters) * SIGMA_SPREAD_COEFFICIENT + SIGMA_SHIFT_COEFFICIENT)
    # Create distributions
    distributions = torch.distributions.Normal(mus, sigmas)
    # Exact samples
    samples = distributions.sample(torch.Size([n_observations,])).t()
    return samples


def generate_data(n_observations: int, k_parameters: int = 2, distribution=Distributions.GAMMA):
    #  Can be replaced by using a proper factory method design pattern but that would involve more code (classes)
    #  and since we're going to put this in a jupyter notebook then I don't know how structured we should
    #  make all of this.
    if distribution == Distributions.GAMMA:
        shapes = torch.abs(torch.randn(k_parameters) * 2 + 2)
        scales = torch.abs(torch.randn(k_parameters) * 1 + 2)
        distributions = torch.distributions.Gamma(shapes, scales)
        samples = distributions.sample(torch.Size([n_observations, ])).t()
        return samples
    elif distribution == Distributions.GAUSSIAN:
        mus = torch.randn(k_parameters) * MU_SPREAD_COEFFICIENT + MU_SHIFT_COEFFICIENT
        sigmas = torch.abs(torch.randn(k_parameters) * SIGMA_SPREAD_COEFFICIENT + SIGMA_SHIFT_COEFFICIENT)
        distributions = torch.distributions.Normal(mus, sigmas)
        samples = distributions.sample(torch.Size([n_observations, ])).t()
        return samples



samples = generate_data(200, 1, distribution=Distributions.GAMMA)
plot_observation_2(samples[0], obs_type=Distributions.GAMMA)

#for index, value in enumerate(samples):
#    plot_observation(value, color=pallet[index % len(pallet)], obs_number=index + 1)
#    plot_observation_2(value, color=pallet[index % len(pallet)], obs_number=index + 1)


#  Genera una matris k x 2 con mu y sigma aleatorios
def init_random_parameters(k_parameters=2):
    mus = torch.randn(k_parameters) * MU_SPREAD_COEFFICIENT + MU_SHIFT_COEFFICIENT
    sigmas = torch.abs(torch.randn(k_parameters) * SIGMA_SPREAD_COEFFICIENT + SIGMA_SHIFT_COEFFICIENT)
    return torch.stack((mus, sigmas), dim=1)


def latex_print_k2_matrix(matrix):
    latex_str = "\\begin{pmatrix}"
    for row in matrix:
        latex_str += " {:.4f} & {:.4f} \\\\".format(row[0].item(), row[1].item())
    latex_str += " \\end{pmatrix}"
    print(latex_str)

