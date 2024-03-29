import torch
from decision_trees import NodeCart
import pytest


data = torch.Tensor([
    [-67, -57, -64, -68, -75, -82, -82, 1],
    [-68, -55, -73, -65, -76, -82, -82, 1],
    [-68, -55, -67, -70, -76, -82, -81, 1],
    [-38, -57, -61, -38, -69, -73, -70, 2],
    [-39, -62, -58, -37, -69, -73, -72, 2],
    [-35, -58, -61, -38, -67, -71, -71, 2],
    [-47, -64, -53, -54, -60, -83, -84, 3],
    [-45, -63, -57, -55, -58, -79, -85, 3],
    [-45, -63, -57, -53, -57, -81, -84, 3],
    [-54, -46, -48, -55, -48, -84, -85, 4],
    [-58, -53, -44, -62, -52, -84, -88, 4],
    [-61, -52, -48, -61, -45, -90, -88, 4],
    [-57, -51, -47, -61, -50, -90, -88, 4],
    [-58, -56, -51, -65, -53, -87, -87, 4],
    [-62, -53, -52, -59, -48, -87, -92, 4],
    [-63, -54, -52, -59, -44, -86, -92, 4],
    [-57, -58, -52, -66, -46, -86, -90, 4]
])


def test_gini_select_best_feature_and_thresh():
    node_tree = NodeCart(gini_entropy_function="GINI")
    min_thresh, min_feature, min_gini = node_tree.select_best_feature_and_thresh(data)
    assert min_thresh == -52.0
    assert min_feature == 2
    assert round(min_gini, 4) == 0.3529


def test_entropy_select_best_feature_and_thresh():
    node_tree = NodeCart(gini_entropy_function="ENTROPY")
    min_thresh, min_feature, min_gini = node_tree.select_best_feature_and_thresh(data)
    assert min_thresh == -52.0
    assert min_feature == 2
    assert round(min_gini, 4) == 0.5816


def test_list_features_select_best_feature_and_thresh():
    node_tree = NodeCart()
    list_features_selected = [2]
    min_thresh, min_feature, min_gini = node_tree.select_best_feature_and_thresh(data, list_features_selected)
    assert min_thresh == -53.0
    assert min_feature == 4
    assert round(min_gini, 4) == 0.3529


def test_full_list_features_select_best_feature_and_thresh():
    node_tree = NodeCart()
    list_features_selected = [0, 1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError) as exc_info:
        node_tree.select_best_feature_and_thresh(data, list_features_selected)
    assert exc_info.value.args[0] == "All features have been selected"


