import torch
from decision_trees import calculate_gini, calculate_total_gini, NodeCart


def test_calculate_gini():
    data = [
        [-64, -56, -61, -66, -71, -82, -81, 1],
        [-68, -57, -61, -65, -71, -85, -85, 1],
        [-42, -53, -62, -38, -66, -65, -69, 2],
        [-44, -55, -61, -41, -66, -72, -68, 2]
    ]
    partition = torch.tensor(data)
    gini = calculate_gini(partition)
    assert gini.item() == 0.5


def test_calculate_total_gini():
    data_left = [
        [-64, -56, -61, -66, -71, -82, -81, 1],
        [-68, -57, -61, -65, -71, -85, -85, 2]
    ]
    data_right = [
        [-42, -53, -62, -38, -66, -65, -69, 1],
        [-44, -55, -61, -41, -66, -72, -68, 1]
    ]
    node_left = NodeCart()
    node_right = NodeCart()
    node_left.data_torch_partition = torch.tensor(data_left)
    node_right.data_torch_partition = torch.tensor(data_right)
    gini_total = calculate_total_gini(node_left, node_right)
    assert gini_total.item() == 0.25
