from decision_trees import NodeCart, read_dataset
from xmldiff import main  # pip install xmldiff


data = read_dataset(csv_name='wifi_localization.txt')


def test_gini_create_with_children():
    node_tree = NodeCart(gini_entropy_function="GINI")
    node_tree.data_torch_partition = data
    node_tree.create_with_children()
    with open("test_gini_create_with_children_tree.xml", "r") as file:
        expected = file.read()
    result = node_tree.to_xml()
    assert main.diff_texts(result, expected) == []


def test_entropy_create_with_children():
    node_tree = NodeCart(gini_entropy_function="ENTROPY")
    node_tree.data_torch_partition = data
    node_tree.create_with_children()
    with open("test_entropy_create_with_children_tree.xml", "r") as file:
        expected = file.read()
    result = node_tree.to_xml()
    assert main.diff_texts(result, expected) == []


