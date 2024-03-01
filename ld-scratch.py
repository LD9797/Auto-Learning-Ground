import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm


MU_SPREAD_COEFFICIENT = 20
MU_SHIFT_COEFFICIENT = 30
SIGMA_SPREAD_COEFFICIENT = 1.5
SIGMA_SHIFT_COEFFICIENT = 3

pallet = ["#1F77B4", "#B41F77", "#77B41F"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


def generate_data_gaussian(n_observations: int, k_parameters: int = 2) -> torch.Tensor:
    mus = torch.randn(k_parameters) * MU_SPREAD_COEFFICIENT + MU_SHIFT_COEFFICIENT
    sigmas = torch.abs(torch.randn(k_parameters) * SIGMA_SPREAD_COEFFICIENT + SIGMA_SHIFT_COEFFICIENT)
    distributions = torch.distributions.Normal(mus, sigmas)
    samples = distributions.sample(torch.Size([n_observations,])).t()
    return samples


def plot_observation(observation: torch.Tensor, show=True, color=pallet[0], obs_number=1):
    fig, ax = plt.subplots()
    x_axis = torch.arange(observation.min(), observation.max(), 0.01)
    ax.hist(observation, density=True, bins=20, alpha=0.5, color=color)
    ax.scatter(observation, torch.zeros(observation.size()), s=6, alpha=0.5, color=color)
    mu = torch.mean(observation)
    sigma = torch.std(observation, unbiased=True)
    label = fr'$\mu={round(mu.item(), 2)},\ \sigma={round(sigma.item(), 2)}$'
    ax.plot(x_axis, norm.pdf(x_axis, mu, sigma), color=color, label=label)
    ax.set_title(f"Generated Gaussian Observation #{obs_number}")
    ax.legend()
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - 0.01, y_max)
    if show:
        plt.show()


def init_random_parameters(k_parameters=2):
    mus = torch.randn(k_parameters) * MU_SPREAD_COEFFICIENT + MU_SHIFT_COEFFICIENT
    sigmas = torch.abs(torch.randn(k_parameters) * SIGMA_SPREAD_COEFFICIENT + SIGMA_SHIFT_COEFFICIENT)
    return torch.stack((mus, sigmas), dim=1)


def calculate_likelihood_gaussian_observation(x_n, mu_k, sigma_k):
    def log_gaussian_function(x, mu, sigma):
        return (-1/2) * torch.log(2 * torch.pi * sigma**2) - ((x-mu)**2/(2 * sigma**2))
    return log_gaussian_function(x_n, mu_k, sigma_k)


def calculate_membership_dataset(observations, parameters):
    mean = parameters[:, 0].unsqueeze(0)
    std = parameters[:, 1].unsqueeze(0)
    observations_expanded = observations.unsqueeze(-1)
    log_likelihoods = calculate_likelihood_gaussian_observation(observations_expanded, mean, std)
    max_values, _ = torch.max(log_likelihoods, dim=-1, keepdim=True)
    one_hot_membership_matrix = (log_likelihoods == max_values).to(torch.float)
    return one_hot_membership_matrix


#  Combinación de la funcion vectorizada y de for de la progra pasada
def recalculate_parameters(x_dataset, membership_data):
    values_per_membership = torch.transpose(membership_data, 0, 1) * x_dataset
    new_parameters = []
    for t_membership in values_per_membership:
        non_zero_mask = t_membership != 0
        t_membership = t_membership[non_zero_mask]
        new_mu = torch.mean(t_membership)
        new_std = torch.std(t_membership)
        new_parameters.append([new_mu, new_std])
    return torch.Tensor(new_parameters)


samples = generate_data_gaussian(200, k_parameters=2)
# Unir los tensores
samples = torch.flatten(samples)
print(samples)
# plot_observation(samples)
random_parameters = init_random_parameters(2)
print(random_parameters)
membership_matrix = calculate_membership_dataset(samples, random_parameters)
recalculated_parameters = recalculate_parameters(samples, membership_matrix)
print(recalculated_parameters)