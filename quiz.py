import torch


def generate_data_gamma(n_observations, k_shape, theta_scale):
    distributions = torch.distributions.Gamma(k_shape, theta_scale)
    sample = distributions.sample_n(n_observations)
    return sample


if __name__ == '__main__':
    k = 0.5
    theta = 2
    n_observations_1 = 1000
    n_observations_2 = 10
    sample_1 = generate_data_gamma(n_observations_1, k, theta)

    sample_1_mean = torch.mean(sample_1)
    sample_1_var = torch.var(sample_1)
    sample_1_k = sample_1_mean ** 2 / sample_1_var
    sample_1_theta = sample_1_var / sample_1_mean

    print(f"N = 1000:\n"
          f"mean = {sample_1_mean}, var = {sample_1_var}\n"
          f"k = mean^2/var, theta = var/mean\n"
          f"k = {sample_1_k}, theta = {sample_1_theta}\n"
          f"Proof:\n"
          f"mean = k*theta = {sample_1_mean} = {sample_1_k} * {sample_1_theta}\n"
          f"{sample_1_mean} = {sample_1_k * sample_1_theta}\n"
          f"var = k*theta^2 = {sample_1_var} = {sample_1_k} * {sample_1_theta}^2\n"
          f"{sample_1_var} = {sample_1_k * sample_1_theta**2}")

    sample_2 = generate_data_gamma(n_observations_2, k, theta)
    sample_2_mean = torch.mean(sample_2)
    sample_2_var = torch.var(sample_2)
    sample_2_k = sample_2_mean ** 2 / sample_2_var
    sample_2_theta = sample_2_var / sample_2_mean

    print(f"N = 10:\n"
          f"mean = {sample_2_mean}, var = {sample_2_var}\n"
          f"k = mean^2/var, theta = var/mean\n"
          f"k = {sample_2_k}, theta = {sample_2_theta}\n"
          f"Proof:\n"
          f"mean = k*theta = {sample_2_mean} = {sample_2_k} * {sample_2_theta}\n"
          f"{sample_2_mean} = {sample_2_k * sample_2_theta}\n"
          f"var = k*theta^2 = {sample_2_var} = {sample_2_k} * {sample_2_theta}^2\n"
          f"{sample_2_var} = {sample_2_k * sample_2_theta ** 2}")