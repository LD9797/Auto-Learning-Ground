import torch
from decision_trees import calculate_gini, calculate_total_gini, calculate_entropy, calculate_total_entropy, NodeCart


data = [
        [-64, -56, -61, -66, -71, -82, -81, 1],
        [-68, -57, -61, -65, -71, -85, -85, 1],
        [-42, -53, -62, -38, -66, -65, -69, 2],
        [-44, -55, -61, -41, -66, -72, -68, 2]
    ]

data_left = [
    [-64, -56, -61, -66, -71, -82, -81, 1],
    [-68, -57, -61, -65, -71, -85, -85, 2]
]

data_right = [
    [-42, -53, -62, -38, -66, -65, -69, 1],
    [-44, -55, -61, -41, -66, -72, -68, 1]
]


def test_calculate_gini():
    partition = torch.tensor(data)
    gini = calculate_gini(partition)
    assert gini.item() == 0.5


def test_calculate_total_gini():
    node_left = NodeCart()
    node_right = NodeCart()
    node_left.data_torch_partition = torch.tensor(data_left)
    node_right.data_torch_partition = torch.tensor(data_right)
    gini_total = calculate_total_gini(node_left, node_right)
    assert gini_total.item() == 0.25


def test_calculate_entropy():
    partition = torch.tensor(data)
    entropy = calculate_entropy(partition)
    assert round(entropy.item(), 3) == 0.693


def test_calculate_total_entropy():
    node_left = NodeCart()
    node_right = NodeCart()
    node_left.data_torch_partition = torch.tensor(data_left)
    node_right.data_torch_partition = torch.tensor(data_right)
    entropy_total = calculate_total_entropy(node_left, node_right)
    assert round(entropy_total.item(), 3) == 0.346


