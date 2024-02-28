import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import math
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
        alpha_hat, loc_hat, beta_hat_inv = gamma.fit(observation, floc=0)
        # beta_hat = 1 / beta_hat_inv  # TODO: Analyze Convert rate to scale if necessary !!
        # TODO https://danielhnyk.cz/fitting-distribution-histogram-using-python/
        # Print estimated parameters
        print(f"Estimated alpha (shape): {alpha_hat}")
        print(f"Estimated beta (scale): {beta_hat_inv}")
        label = ""
        ax.plot(x_axis, gamma.pdf(x_axis, alpha_hat, loc_hat, beta_hat_inv), color=color, label=label)
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
    samples = distributions.sample(torch.Size([n_observations, ])).t()
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


# for index, value in enumerate(samples):
#    plot_observation(value, color=pallet[index % len(pallet)], obs_number=index + 1)
#    plot_observation_2(value, color=pallet[index % len(pallet)], obs_number=index + 1)


#  Genera una matriz k x 2 con mu y sigma aleatorios
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


def calculate_likelihood_gaussian_dataset(x_n, mu_k, sigma_k):
    n = x_n.shape[0]
    return (-(n / 2) * torch.log(torch.tensor(2 * torch.pi)) - n * torch.log(sigma_k) - (
                1 / (2 * sigma_k ** 2)) * torch.sum((x_n - mu_k) ** 2))


def calculate_likelihood_dataset_alt(samples: torch.tensor, parameters: torch.tensor, k_parameters: int):
    mu_k = parameters[:, 0][:, None]
    sigma_k = parameters[:, 1][:, None]
    likelihood = torch.log(1/(torch.sqrt(2 * torch.pi * sigma_k**2))) + torch.log(torch.exp((-1/2) * ((samples.repeat(k_parameters, 1) - mu_k) / sigma_k)**2))
    #torch.nan_to_num((1 / torch.sqrt(2 * math.pi * std**2)) * math.e**(-(1/2) * ((samples.repeat(2, 1) - mean) / std)**2))
    return likelihood


def calculate_likelihood_gaussian_observation(x, mu_k, sigma_k):
    return (torch.log(1/(torch.sqrt(2 * torch.pi * sigma_k**2))) +
            torch.log(torch.exp((-1/2) * ((x - mu_k) / sigma_k)**2)))


def test_calculate_likelihood_gaussian_observation():
    samples = generate_data_gaussian(100, 1)
    sample = samples[0]
    # TODO justificar calculo de metodos
    real_mu = torch.mean(sample)
    real_sigma = torch.std(sample, unbiased=True)
    random_params = init_random_parameters(1)
    fake_mu = random_params[0][0]
    fake_sigma = random_params[0][1]
    fake_likelihood = calculate_likelihood_gaussian_dataset(sample, fake_mu, fake_sigma)
    real_likelihood = calculate_likelihood_gaussian_dataset(sample, real_mu, real_sigma)
    assert real_likelihood > fake_likelihood


#  Calcula la pertenencia de cada observación a los parámetros
#  Se utiliza one hot vector
def calculate_membership_dataset(x_dataset, parameters_matrix):
    likelihood_matrix = []
    for dataset in x_dataset:
        for data in dataset:
            data_likelihood = []
            for matrix in parameters_matrix:
                mu = matrix[0]
                sigma = matrix[1]
                likelihood = calculate_likelihood_gaussian_observation(data, mu, sigma)
                data_likelihood.append(likelihood)
            for index in range(len(data_likelihood)):
                data_likelihood[index] = 0 if data_likelihood[index] != max(data_likelihood) else 1
            likelihood_matrix.append(data_likelihood)
    likelihood_matrix = torch.tensor(likelihood_matrix)
    return likelihood_matrix


samples = generate_data_gaussian(20, 2)
parameters = init_random_parameters(2)

# likelihood = calculate_likelihood_dataset_alt(samples, parameters)

likelihood = calculate_likelihood_dataset_alt(samples[0], parameters, 2)
pass


def calculate_membership_gaussian_dataset(x_n, mu_k, sigma_k):
    original = calculate_likelihood_gaussian_dataset(x_n, mu_k, sigma_k)
    transpose_o = torch.t(original)
    max_values = torch.amax(transpose_o, 1)
    return torch.where(original == max_values, 1.0, 0.0)


def test_calculate_membership_gaussian_dataset():
    samples = generate_data_gaussian(100, 1)
    sample = samples[0]
    # TODO justificar calculo de metodos
    real_mu = torch.mean(sample)
    real_sigma = torch.std(sample, unbiased=True)
    random_params = init_random_parameters(1)
    fake_mu = random_params[0][0]
    fake_sigma = random_params[0][1]
    membership1 = calculate_membership_gaussian_dataset(sample, fake_mu, fake_sigma)
    membership2 = calculate_membership_gaussian_dataset(sample, real_mu, real_sigma)

    print('Membership1 ->\n', membership1)
    print('Membership2 ->\n', membership2)
