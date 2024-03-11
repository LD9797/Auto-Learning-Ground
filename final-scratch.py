import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gamma
# Based on https://pypi.org/project/torch-kmeans/
from torch_kmeans import KMeans

torch.set_printoptions(sci_mode=False)


# Constants
BELL_Y_OFFSET = 0.005
BELL_DISTANCE_FACTOR = 0.2

MU_SHIFT_COEFFICIENT = 30
MU_SPREAD_COEFFICIENT = 10

SIGMA_SHIFT_COEFFICIENT = 3
SIGMA_SPREAD_COEFFICIENT_START = 1.0
SIGMA_SPREAD_COEFFICIENT_END = 10.0

PALETTE = ["#9E0000", "#4F9E00", "#009E9E"]


def generate_data(n_observations, parameters):
    mus = parameters[:, 0][:, None].squeeze(1)
    sigmas = parameters[:, 1][:, None].squeeze(1)
    distributions = torch.distributions.Normal(mus, sigmas)
    samples = distributions.sample(torch.Size([n_observations,])).t()
    return samples


def init_original_parameters(k_parameters=2):
    # Creates a set of mus that starts on MU_SHIFT_COEFFICIENT and ends on MU_SHIFT_COEFFICIENT * k_parameters
    # with step MU_SHIFT_COEFFICIENT. A random number between 0 and MU_SPREAD_COEFFICIENT is added to each
    # value in the range
    mu_range = torch.range(MU_SHIFT_COEFFICIENT, MU_SHIFT_COEFFICIENT * k_parameters, MU_SHIFT_COEFFICIENT)
    mus = torch.rand(k_parameters) * MU_SPREAD_COEFFICIENT + mu_range

    # Creates a set of sigmas between SIGMA_SHIFT_COEFFICIENT and SIGMA_SPREAD_COEFFICIENT_END
    sigma_spread = torch.rand(1) * (
                SIGMA_SPREAD_COEFFICIENT_END - SIGMA_SPREAD_COEFFICIENT_START) + SIGMA_SPREAD_COEFFICIENT_START
    sigmas = torch.rand(k_parameters) * sigma_spread + SIGMA_SHIFT_COEFFICIENT

    return torch.stack((mus, sigmas), dim=1)


def plot_observation(observation, show=True, color=None, title="", show_hist=True, fig=None, ax=None,
                    show_curve=True, y_adjustment=True):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    if show_hist:
        ax.hist(observation, density=True, bins=20, alpha=0.5, color=color)
    ax.scatter(observation, torch.zeros(observation.size()), s=8, alpha=0.5, color=color)
    if show_curve:
        x_axis = torch.arange(observation.min().item(), observation.max().item(), 0.01)
        mu = torch.mean(observation)
        sigma = torch.std(observation, unbiased=True)
        label = fr'$\mu={round(mu.item(), 2)},\ \sigma={round(sigma.item(), 2)}$'
        ax.plot(x_axis, norm.pdf(x_axis, mu, sigma), color=color, label=label)
        ax.legend()
    if title != "":
        ax.set_title(title)
    if y_adjustment:
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min - 0.01, y_max)
    if show:
        plt.show()


def init_random_parameters(k_parameters=2):
    # Creates a range of mus between MU_SHIFT_COEFFICIENT and MU_SHIFT_COEFFICIENT * k_parameters
    mus = torch.rand(k_parameters) * (MU_SHIFT_COEFFICIENT * k_parameters - MU_SHIFT_COEFFICIENT) + MU_SHIFT_COEFFICIENT
    # Creates a range of sigmas between SIGMA_SPREAD_COEFFICIENT_START and SIGMA_SPREAD_COEFFICIENT_END
    sigmas = (torch.rand(k_parameters) * (SIGMA_SPREAD_COEFFICIENT_END - SIGMA_SPREAD_COEFFICIENT_START) +
              SIGMA_SPREAD_COEFFICIENT_START)
    return torch.stack((mus, sigmas), dim=1)


# TODO: cambiar a calculate_likelihood_gaussian_dataset y hacer prueba unitaria
def calculate_likelihood_gaussian_observation(x_n, mu_k, sigma_k):
    def log_gaussian_function(x, mu, sigma):
        return (-1/2) * torch.log(2 * torch.pi * sigma**2) - ((x-mu)**2/(2 * sigma**2))
    return log_gaussian_function(x_n, mu_k, sigma_k)


# TODO: Prueba unitaria
def calculate_membership_dataset(observations, parameters):
    mean = parameters[:, 0].unsqueeze(0)
    std = parameters[:, 1].unsqueeze(0)
    observations_expanded = observations.unsqueeze(-1)
    log_likelihoods = calculate_likelihood_gaussian_observation(observations_expanded, mean, std)
    max_values, _ = torch.max(log_likelihoods, dim=-1, keepdim=True)
    one_hot_membership_matrix = (log_likelihoods == max_values).to(torch.float)
    return one_hot_membership_matrix


def recalculate_parameters(x_dataset, membership_data):
    values_per_membership = torch.transpose(membership_data, 0, 1) * x_dataset
    new_parameters = []
    for t_membership in values_per_membership:
        non_zero_mask = t_membership != 0
        t_membership = t_membership[non_zero_mask]
        new_mu = torch.mean(t_membership)
        new_std = torch.std(t_membership)
        if new_mu.item() != new_mu.item() or new_std.item() != new_std.item():  # if nan
            params = init_random_parameters(1)
            new_mu = params[0][0]
            new_std = params[0][1]
        new_parameters.append([new_mu, new_std])
    return torch.Tensor(new_parameters)


def plot_gaussian_distribution_and_observations(distribution_parameters, observations, title=""):
    """ What it does

    Summary

    Args:

    Returns:
    """
    fig, ax = plt.subplots()

    if observations.dim() > 1:
        for index, sample in enumerate(observations):
            plot_observation(sample, color=PALETTE[index % len(PALETTE)], show=False, show_hist=False, fig=fig, ax=ax,
                             y_adjustment=False, show_curve=False)
    else:
        plot_observation(observations, show=False, show_hist=False, fig=fig, ax=ax, y_adjustment=False)

    for index, parameters in enumerate(distribution_parameters):
        mu = parameters[0]
        sigma = parameters[1]
        min_value = torch.min(observations)
        max_value = torch.max(observations)
        dist = (max_value - min_value) * BELL_DISTANCE_FACTOR
        x_axis = torch.arange(min_value.item() - dist, max_value.item() + dist)
        ax.plot(x_axis, norm.pdf(x_axis, mu, sigma) + BELL_Y_OFFSET,
                label=r'$\mu_' + str(index + 1) + r'=' + str(round(mu.item(), 2)) +
                      r',\ \sigma_' + str(index + 1) + '=' + str(round(sigma.item(), 2)) + r'$',
                color=PALETTE[index % len(PALETTE)])

    if title != "":
        ax.set_title(title)

    plt.legend()
    plt.show()


def expectation_maximization(samples, iterations=5, distributions_to_plot=3, run_number=1, heuristic=False):
    parameters = init_random_parameters(samples.size(0)) if not heuristic else heuristic_improvement(samples)
    plot_gaussian_distribution_and_observations(parameters, samples, title=f"Iteration #0 | Run #{run_number}")
    plots_to_show = torch.randperm(iterations - 1)[:distributions_to_plot] + 1
    for iteration in range(1, iterations + 1):
        membership_data = calculate_membership_dataset(torch.flatten(samples), parameters)
        parameters = recalculate_parameters(torch.flatten(samples), membership_data)
        if iteration in plots_to_show:
            plot_gaussian_distribution_and_observations(parameters, samples,
                                                        title=f'Iteration #{iteration} | Run #{run_number}')
    plot_gaussian_distribution_and_observations(parameters, samples,
                                                title=f'Final Iteration #{iterations} | Run #{run_number}')
    return parameters


def run_algorithm(heuristic=False):
    initial_parameters = init_original_parameters(3)
    gaussian_samples = generate_data(200, initial_parameters)
    final_parameters = []
    for run in range(5):
        result_parameters = expectation_maximization(gaussian_samples, run_number=run, heuristic=heuristic)
        final_parameters.append(result_parameters)


# run_algorithm()


def heuristic_improvement(test_data):
    k = test_data.size(0)
    model = KMeans(n_clusters=k)
    test_data = test_data.unsqueeze(2)
    result = model(test_data)
    # Mu estimation
    centroides = result.centers
    centroides = centroides.flatten()
    centroides = centroides[::k]
    centroides = centroides.reshape(k, 1)
    # Sigma estimation
    inertia = result.inertia
    varianza = torch.zeros(k, 1)
    for idx, elem in enumerate(inertia):
        varianza[idx] = torch.sqrt(elem / test_data.size(1))
    new_params = torch.cat((centroides, varianza), dim=1)
    return new_params


run_algorithm(heuristic=True)

# Gamma

# 1. Ajustar la function generate_data con los parametros k y theta de gamma.
# 2. Modificar la funcion para la observacion de la curva gamma.
# 3. Modificar la funcion init_random_parameters para generar los k y theta correspondientes a la f gamma.
# 4. Programar la funcion gamma para el calculo de la verosmilitud.
# 5. Adaptar recalculate_parameters y calculate_membership_dataset con la funcion gamma

SHAPE_MEAN = 2
SHAPE_STD = 2
SCALE_MEAN = 2
SCALE_STD = 1
LOC_STEP = 3
LOC_MAX_RANGE = 30


def generate_data_gamma(n_observations: int, k_parameters: int = 2):
    shapes = torch.abs(torch.randn(k_parameters)) * SHAPE_STD + SHAPE_MEAN
    scales = torch.abs(torch.randn(k_parameters)) * SCALE_STD + SCALE_MEAN
    distributions = torch.distributions.Gamma(shapes, scales)
    samples = distributions.sample(torch.Size([n_observations, ])).t()
    loc_range = torch.range(0, LOC_MAX_RANGE, LOC_STEP)[0:k_parameters].unsqueeze(-1)
    samples += loc_range
    return samples


def plot_observation_gamma(observation: torch.Tensor, show=True, color=None, title="", show_hist=True, show_curve=True,
                           fig=None, ax=None, y_adjustment=True):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.scatter(observation, torch.zeros(observation.size()), s=6, alpha=0.5, color=color)
    if show_hist:
        ax.hist(observation, density=True, bins=20, alpha=0.5, color=color)
    if show_curve:
        # loc -> shift the distribution along the x-axis
        shape, loc, scale = gamma.fit(observation, floc=0)
        label = fr"$k={round(shape, 2)},\ \theta={round(scale, 2)}$"
        x_axis = torch.arange(observation.min().item(), observation.max().item(), 0.01)
        ax.plot(x_axis, gamma.pdf(x_axis, shape, loc, scale), color=color, label=label)
        ax.legend()
    if title != "":
        ax.set_title(title)
    if y_adjustment:
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min - 0.01, y_max)
    if show:
        plt.show()


def plot_gamma_distribution_and_observations(distribution_parameters, observations, title=""):
    """ What it does

    Summary

    Args:

    Returns:
    """
    fig, ax = plt.subplots()

    if observations.dim() > 1:
        for index, sample in enumerate(observations):
            plot_observation_gamma(sample, color=PALETTE[index % len(PALETTE)], show=False, show_hist=False, fig=fig,
                                   ax=ax, y_adjustment=False, show_curve=False)
    else:
        plot_observation_gamma(observations, show=False, show_hist=False, fig=fig, ax=ax, y_adjustment=False)

    for index, parameters in enumerate(distribution_parameters):
        shape = parameters[0]
        scale = parameters[1]
        min_value = torch.min(observations)
        max_value = torch.max(observations)
        dist = (max_value - min_value) * BELL_DISTANCE_FACTOR
        x_axis = torch.arange(min_value.item() - dist, max_value.item() + dist)
        ax.plot(x_axis, gamma.pdf(x_axis, shape, 0, scale) + BELL_Y_OFFSET,
                label=r'$k_' + str(index + 1) + r'=' + str(round(shape.item(), 2)) +
                      r',\ \theta_' + str(index + 1) + '=' + str(round(scale.item(), 2)) + r'$',
                color=PALETTE[index % len(PALETTE)])
    if title != "":
        ax.set_title(title)

    plt.legend()
    plt.show()


def calculate_likelihood_gamma_observation(x_n, form_k, scale_th):
    def gamma_function(x, form, scale):
        factorial_part = torch.exp(torch.lgamma(form))
        return (x**(form - 1) * torch.exp(-x/scale)) / (factorial_part * scale**form)
    return gamma_function(x_n, form_k, scale_th)


K_START = 1
K_END = 10
SCALE_START = 1
SCALE_END = 5


def init_random_parameters_gamma(k_parameters=2, n_observations=200):
    shapes = (torch.rand(k_parameters) * (K_END - K_START) + K_START)
    scales = (torch.rand(k_parameters) * (SCALE_END - SCALE_START) + SCALE_START)
    distributions = torch.distributions.Gamma(shapes, scales)
    samples = distributions.sample(torch.Size([n_observations, ])).t()
    shapes = []
    scales = []
    for sample in samples:
        shape, loc, scale = gamma.fit(sample, floc=0)
        shapes.append(shape)
        scales.append(scale)
    return torch.stack((torch.tensor(shapes), torch.tensor(scales)), dim=1)


def recalculate_parameters_gamma(x_dataset, membership_data):
    values_per_membership = torch.transpose(membership_data, 0, 1) * x_dataset
    new_parameters = []
    for t_membership in values_per_membership:
        non_zero_mask = t_membership != 0
        t_membership = t_membership[non_zero_mask]
        if t_membership.sum() == 0 or t_membership.size()[0] == 1:
            params = init_random_parameters_gamma(1)
            new_shape = params[0][0]
            new_scale = params[0][1]
        else:
            new_shape, loc, new_scale = gamma.fit(t_membership, floc=0)
        new_parameters.append([new_shape, new_scale])
    return torch.Tensor(new_parameters)


def calculate_membership_dataset_gamma(observations, parameters):
    shape = parameters[:, 0].unsqueeze(0)
    scale = parameters[:, 1].unsqueeze(0)
    observations_expanded = observations.unsqueeze(-1)
    log_likelihoods = calculate_likelihood_gamma_observation(observations_expanded, shape, scale)
    max_values, _ = torch.max(log_likelihoods, dim=-1, keepdim=True)
    one_hot_membership_matrix = (log_likelihoods == max_values).to(torch.float)
    return one_hot_membership_matrix


def expectation_maximization_gamma(samples, iterations=5, distributions_to_plot=3, run_number=1):
    parameters = init_random_parameters_gamma(samples.size(0))
    plot_gamma_distribution_and_observations(parameters, samples, title=f"Iteration #0 | Run #{run_number}")
    plots_to_show = torch.randperm(iterations - 1)[:distributions_to_plot] + 1
    for iteration in range(1, iterations + 1):
        membership_data = calculate_membership_dataset_gamma(torch.flatten(samples), parameters)
        parameters = recalculate_parameters_gamma(torch.flatten(samples), membership_data)
        if iteration in plots_to_show:
            pass
            plot_gamma_distribution_and_observations(parameters, samples,
                                                        title=f'Iteration #{iteration} | Run #{run_number}')
    plot_gamma_distribution_and_observations(parameters, samples,
                                                title=f'Final Iteration #{iterations} | Run #{run_number}')
    return parameters


def run_algorithm_gamma():
    gamma_samples = generate_data_gamma(200, 3)
    # TODO Calculate original parameters from generated data
    final_parameters = []
    for run in range(1):
        result_parameters = expectation_maximization_gamma(gamma_samples, run_number=run, iterations=50)
        final_parameters.append(result_parameters)


#  run_algorithm_gamma()

