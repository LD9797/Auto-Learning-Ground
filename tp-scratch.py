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


def plot_observation(observation, show=True, color=pallet[0], obs_number=1):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    mu = torch.mean(observation)
    sigma = torch.std(observation, unbiased=True)
    x_axis = torch.arange(observation.min(), observation.max(), 0.01)
    ax1.scatter(observation, torch.zeros(observation.size()), s=3, alpha=0.5, color=color)
    label = fr'$\mu={round(mu.item(), 2)},\ \sigma={round(sigma.item(), 2)}$'
    ax1.plot(x_axis, norm.pdf(x_axis, mu, sigma), color=color, label=label)
    ax1.legend()
    ax2.hist(observation, bins=20, alpha=0.5, color=color)
    ax1.set_title(f"Generated Observation #{obs_number}")
    if show:
        plt.show()


def plot_observation_2(observation, show=True, color=pallet[0], obs_number=1):
    fig, ax = plt.subplots()
    mu = torch.mean(observation)
    sigma = torch.std(observation, unbiased=True)
    x_axis = torch.arange(observation.min(), observation.max(), 0.01)
    # Histogram
    ax.hist(observation, density=True, bins=20, alpha=0.5, color=color)
    # Scatter Dots
    ax.scatter(observation, torch.zeros(observation.size()), s=6, alpha=0.5, color=color)
    # Gaussian Curve
    label = fr'$\mu={round(mu.item(), 2)},\ \sigma={round(sigma.item(), 2)}$'
    ax.plot(x_axis, norm.pdf(x_axis, mu, sigma), color=color, label=label)
    ax.set_title(f"Generated Observation #{obs_number}")
    ax.legend()
    # Adjust y-axis
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - 0.01, y_max)
    if show:
        plt.show()


def generate_data(n_observations: int, k_parameters: int = 2) -> torch.Tensor:
    mus = torch.randn(k_parameters) * MU_SPREAD_COEFFICIENT + MU_SHIFT_COEFFICIENT
    sigmas = torch.abs(torch.randn(k_parameters) * SIGMA_SPREAD_COEFFICIENT + SIGMA_SHIFT_COEFFICIENT)
    # Create distributions
    distributions = torch.distributions.Normal(mus, sigmas)
    # Exact samples
    samples = distributions.sample(torch.Size([n_observations,])).t()
    return samples


#samples = generate_data(200, 1)

#for index, value in enumerate(samples):
#    plot_observation(value, color=pallet[index % len(pallet)], obs_number=index + 1)
#    plot_observation_2(value, color=pallet[index % len(pallet)], obs_number=index + 1)


#  Genera una matris k x 2 con mu y sigma aleatorios
def init_random_parameters(k_parameters=2):
    mus = torch.randn(k_parameters) * MU_SPREAD_COEFFICIENT + MU_SHIFT_COEFFICIENT
    sigmas = torch.abs(torch.randn(k_parameters) * SIGMA_SPREAD_COEFFICIENT + SIGMA_SHIFT_COEFFICIENT)
    return torch.stack((mus, sigmas), dim=1)


x = init_random_parameters()


def pretty_print_k2_matrix(matrix):
    latex_str = "\\begin{pmatrix}"
    for row in x:
        latex_str += " {:.4f} & {:.4f} \\\\".format(row[0].item(), row[1].item())
    latex_str += " \\end{pmatrix}"
    print(latex_str)

