import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch_kmeans import KMeans
torch.set_printoptions(sci_mode=False)



MU_SPREAD_COEFFICIENT = 10
MU_SHIFT_COEFFICIENT = 30
SIGMA_SPREAD_COEFFICIENT = 1.5
SIGMA_SHIFT_COEFFICIENT = 3

pallet = ["#9E0000", "#4F9E00", "#009E9E"]

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using {device}")


def init_random_parameters(k_parameters=2, initial_parameters=False):
    if initial_parameters:
        # Para generar una segregación más visible entre los datos iniciales
        sigma_range = torch.range(1, 5, 0.5)
        sigma_rand_index = torch.randint(0, len(sigma_range), (1,)).item()
        sigma_spread = sigma_range[sigma_rand_index] + torch.rand(1).item()
        mu_range = torch.range(MU_SHIFT_COEFFICIENT, MU_SHIFT_COEFFICIENT * k_parameters, MU_SHIFT_COEFFICIENT)
        mus = torch.abs(torch.randn(k_parameters)) * MU_SPREAD_COEFFICIENT + mu_range
        sigmas = torch.abs(torch.randn(k_parameters) * sigma_spread) + SIGMA_SHIFT_COEFFICIENT
    else:
        # Para generar una inicialización más aleatoria sin considerar tanto la segregación de los datos
        mus = torch.randn(k_parameters) * MU_SPREAD_COEFFICIENT + MU_SHIFT_COEFFICIENT
        sigmas = torch.abs(torch.randn(k_parameters) * SIGMA_SPREAD_COEFFICIENT + SIGMA_SHIFT_COEFFICIENT)
    return torch.stack((mus, sigmas), dim=1)


def generate_data_gaussian(n_observations: int, k_parameters: int = 2) -> torch.Tensor:
    original_parameters = init_random_parameters(k_parameters, initial_parameters=True)
    distributions = torch.distributions.Normal(original_parameters[:, 0][:, None].squeeze(1),
                                               original_parameters[:, 1][:, None].squeeze(1))
    samples = distributions.sample(torch.Size([n_observations,])).t()
    return samples


def plot_observation(observation: torch.Tensor, show=True, color=pallet[0], title="", show_hist=True, show_curve=True,
                     fig=None, ax=None, y_adjustment=True):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    x_axis = torch.arange(observation.min().item(), observation.max().item(), 0.01)
    if show_hist:
        ax.hist(observation, density=True, bins=20, alpha=0.5, color=color)
    ax.scatter(observation, torch.zeros(observation.size()), s=6, alpha=0.5, color=color)
    mu = torch.mean(observation)
    sigma = torch.std(observation, unbiased=True)
    label = fr'$\mu={round(mu.item(), 2)},\ \sigma={round(sigma.item(), 2)}$'
    if show_curve:
        ax.plot(x_axis, norm.pdf(x_axis, mu, sigma), color=color, label=label)
        ax.legend()
    if title != "":
        ax.set_title(title)
    if y_adjustment:
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min - 0.01, y_max)
    if show:
        plt.show()


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


#  Plot random distributions alongside the generated observations
def plot_gaussian_distribution_and_observations(distribution_parameters, observations, show=False):
    fig, ax = plt.subplots()
    if observations.dim() > 1:
        for index, sample in enumerate(observations):
            plot_observation(sample, color=pallet[index % 3], show=False, show_hist=False, show_curve=False, fig=fig,
                             ax=ax, y_adjustment=False)
    else:
        plot_observation(observations, show=False, show_hist=False, show_curve=False, fig=fig, ax=ax,
                         y_adjustment=False)
    for index, parameters in enumerate(distribution_parameters):
        mu = parameters[0]
        sigma = parameters[1]
        x_axis = torch.arange(mu / 2, mu * 2, 0.01)
        plt.plot(x_axis.numpy(), norm.pdf(x_axis.numpy(), mu.numpy(), sigma.numpy()),
                 label=r'$\mu_' + str(index + 1) + r'=' + str(round(mu.item(), 2)) +
                       r',\ \sigma_' + str(index + 1) + '=' + str(round(sigma.item(), 2)) + r'$')
    if show:
        plt.legend()
        plt.show()


def expectation_maximization(observations=200, k_parameters=2, iterations=5):
    my_data = generate_data_gaussian(observations, k_parameters)
    parameters = init_random_parameters(k_parameters)
    plot_gaussian_distribution_and_observations(parameters, my_data, show=True)
    for iteration in range(iterations):
        membership_data = calculate_membership_dataset(torch.flatten(my_data), parameters)
        parameters = recalculate_parameters(torch.flatten(my_data), membership_data)
        plot_gaussian_distribution_and_observations(parameters, my_data, show=True)


expectation_maximization()

def heuristic_improvement(test_data,k=2):
   
    model = KMeans(n_clusters=k)

    test_data = test_data.unsqueeze(2)
    result = model(test_data)

    #print("Centers: ", result.centers)
    #print("Inertia: ",result.inertia)
        
    #Mu estimation
    centroides = result.centers
    centroides = centroides.flatten()
    #print("Tensor de centroides flat:", centroides)
    centroides = centroides[::k]
    #print("Centroids pares:", centroides)
    centroides = centroides.reshape(k,1)
    #print("Tensor de centroides ajustados:", centroides)
        
    # Sigma estimation
    inertia = result.inertia
    varianza = torch.zeros(k,1)

    for idx, elem in enumerate(inertia):
        varianza[idx]= test_data.size(1) / elem
        #print("Indice", test_data.size(1))
        #print("Elemento", elem)

    #varianza = torch.tensor(varianza)
    #varianza = varianza.reshape(k,1)
    #print("Tensor de varianza:", varianza)
    #print("Tensor de varianza:", varianza.size())

    new_params = torch.cat((centroides, varianza),dim=1)

    return new_params

sample_data = generate_data_gaussian(20,3)
heuristic_improvement(sample_data,sample_data.size(0))