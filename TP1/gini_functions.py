import torch


def calculate_gini(data_partition_torch, classes):
    def calculate_gini_impurity(partition):
        size = partition.shape[0]
        if size == 0:  # To handle the division by zero
            return torch.tensor(0)
        length = partition.shape[1] - 1
        proportions = torch.Tensor([((partition[:, length] == label).sum() / size) ** 2 for label in classes])
        return 1 - torch.sum(proportions)
    return calculate_gini_impurity(data_partition_torch)


def calculate_total_gini(node_left, node_right, classes):
    size_left = node_left.data_torch_partition.shape[0]
    size_right = node_right.data_torch_partition.shape[0]
    size_total = size_left + size_right
    gini_left = calculate_gini(node_left.data_torch_partition, classes)
    gini_right = calculate_gini(node_right.data_torch_partition, classes)
    gini_total = (size_left / size_total) * gini_left + (size_right / size_total) * gini_right
    return gini_total


def calculate_entropy(data_partition_torch):
    def calculate_entropy_impurity(partition):
        size = partition.shape[0]
        if size == 0:  # To handle the division by zero
            return torch.tensor(0)
        length = partition.shape[1] - 1
        epsilon = torch.tensor(0.0001)
        _, counts = partition[:, length].unique(return_counts=True)
        probabilities = (counts / size) + epsilon
        probabilities = probabilities * torch.log(probabilities)
        entropy = - probabilities.sum()
        return entropy
    return calculate_entropy_impurity(data_partition_torch)


def calculate_total_entropy(node_left, node_right):
    size_left = node_left.data_torch_partition.shape[0]
    size_right = node_right.data_torch_partition.shape[0]
    size_total = size_left + size_right
    entropy_left = calculate_entropy(node_left.data_torch_partition)
    entropy_right = calculate_entropy(node_right.data_torch_partition)
    entropy_total = (size_left / size_total) * entropy_left + (size_right / size_total) * entropy_right
    return entropy_total

