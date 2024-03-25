import torch
import pandas
import numpy as np

CLASSES = [1, 2, 3, 4]


def read_dataset(csv_name='wifi_localization.txt'):
    """

    :param csv_name:
    :return:
    """
    data_frame = pandas.read_table(csv_name, sep=r'\s+', names=('A', 'B', 'C', 'D', 'E', 'F', 'G', 'ROOM'),
                                   dtype={'A': np.int64, 'B': np.float64, 'C': np.float64, 'D': np.float64,
                                          'E': np.float64, 'F': np.float64, 'G': np.float64, 'ROOM': np.float64})
    # targets_torch = torch.tensor(data_frame['ROOM'].values)
    dataset_torch = torch.tensor(data_frame.values)
    return dataset_torch


class NodeCart:

    def __init__(self, num_classes=4, ref_cart=None, current_depth=0):
        """

        :param num_classes:
        :param ref_cart:
        :param current_depth:
        """
        self.ref_CART = ref_cart
        self.threshold_value = 0  # Umbral
        self.feature_num = 0
        self.node_right = None
        self.node_left = None
        self.data_torch_partition = None  # Referencia a la partición del dato
        self.gini = 0  # O Entropia. Funcion numerica a utilizar cuando se construya el arbol.
        self.dominant_class = None  # Clase con mayor cantidad de observaciones en esa particion.
        self.accuracy_dominant_class = None  # Tasa de aciertos de esa clase dominante
        # self.num_classes = num_classes
        self.current_depth = current_depth  # Profundidad

    def to_xml(self, current_str=""):
        """
        Recursive function to write the node content to an xml formatted string
        param current_str : the xml content so far in the whole tree
        return the string with the node content
        """
        str_node = (f"<node>"
                    f"<thresh>{self.threshold_value}</thresh>"
                    f"<feature>{self.feature_num}</feature>"
                    f"<depth>{self.current_depth}</depth>"
                    f"<gini>{self.gini}</gini>")
        if not self.node_right:
            str_left = self.node_right.to_xml(current_str)
            str_node += str_left
        if not self.node_left:
            str_right = self.node_left.to_xml(current_str)
            str_node += str_right
        if self.is_leaf():
            str_node += (f"<dominant_class>{self.dominant_class}</dominant_class>"
                         f"<acc_dominant_class>{self.accuracy_dominant_class}</acc_dominant_class>")
        str_node += "</node>"
        return str_node

    def is_leaf(self):
        """
        Checks whether the node is a leaf
        :return:
        """
        return self.node_left and self.node_right

    def create_with_children(self, data_torch, current_depth, list_selected_features=None, min_gini=0.000001):
        """
        Creates a node by selecting the best feature and threshold, and if needed, creating its children
        param data_torch: dataset with the current partition to deal with in the node
        param current_depth: depth counter for the node
        param list_selected_features: list of selected features so far for the CART building process
        param min_gini: hyperparameter selected by the user defining the minimum tolerated Gini coefficient for a node
        return the list of selected features so far
        """

        if list_selected_features is None:
            list_selected_features = []
        return list_selected_features

    def select_best_feature_and_thresh(self, data_torch, list_features_selected=None, num_classes=4):
        """
        Selects the best feature and threshold that minimizes the Gini coefficient
        param data_torch: dataset partition to analyze
        param list_features_selected list of features selected so far, thus must be ignored
        param num_classes: number of K classes to discriminate from
        return min_thresh, min_feature, min_gini found for the dataset partition when
        selecting the found feature and threshold
        """

        # TODO
        # return selected cut
        if list_features_selected is None:
            list_features_selected = []
        # return (min_thresh, min_feature, min_gini)
        pass

    def calculate_gini(self, data_partition_torch, num_classes=4):
        """
        Calculates the Gini coefficient for a given partition with the given number of classes
        param data_partition_torch: current dataset partition as a tensor
        param num_classes: K number of classes to discriminate from
        returns the calculated Gini coefficient
        """
        # TODO

        # return gini
        pass

    def calculate_entropy(self, data_partition_torch, num_classes=4):
        """
        Calculates the Gini coefficient for a given partition with the given number of classes
        param data_partition_torch: current dataset partition as a tensor
        param num_classes: K number of classes to discriminate from
        returns the calculated Gini coefficient
        """
        # TODO

        # return gini
        pass

    def evaluate_node(self, input_torch):
        """
        Evaluates an input observation within the node.
        If is not a leaf node, send it to the corresponding node
        return predicted label
        """
        feature_val_input = input_torch[self.feature_num]
        if self.is_leaf():
            return self.dominant_class
        elif feature_val_input < self.threshold_value:
            return self.node_left.evaluate_node(input_torch)
        else:
            return self.node_right.evaluate_node(input_torch)


class CART:  # Este es el arbol
    def __init__(self, dataset_torch, max_cart_depth, min_observations=2):
        """
        CART has only one root node
        """
        # min observations per node
        self.min_observations = min_observations # Para evitar sobre ajuste, se definen algunos valores para permitir controlar la complejidad
        # Si el arbol se hace muy complejo puede llegar a ser uno que se sobreajusta a los datos.
        self.root = NodeCart(num_classes=4, ref_CART=self, current_depth=0)
        self.max_CART_depth = max_cart_depth
        self.list_selected_features = []

    def get_root(self):
        """
        Gets tree root
        """
        return self.root

    def get_min_observations(self):
        """
        return min observations per node
        """
        return self.min_observations

    def get_max_depth(self):
        """
        Gets the selected max depth of the tree
        """
        return self.max_CART_depth

    def build_cart(self, data_torch):
        """
        Build CART from root
        """
        self.list_selected_features = self.root.create_with_children(data_torch, current_depth=0)

    def to_xml(self, xml_file_name):
        """
        write Xml file with tree content
        """
        str_nodes = self.root.to_xml()
        with open(xml_file_name, 'w') as file:
            file.write(str_nodes)
        return str_nodes

    def evaluate_input(self, input_torch):
        """
        Evaluate a specific input in the tree and get the predicted class
        """
        return self.root.evaluate_node(input_torch)


def train_cart(dataset_torch, name_xml="", max_cart_depth=3, min_obs_per_leaf=2):
    """
    Train CART model
    """
    tree = CART(dataset_torch=dataset_torch, max_cart_depth=max_cart_depth, min_observations=min_obs_per_leaf)
    tree.build_cart(dataset_torch)
    if name_xml:
        tree.to_xml(name_xml)
    return tree


def test_cart(tree, testset_torch):
    """
    Test a previously built CART
    """
    # TODO, use tree.evaluate_input(current_observation) for this
    # return accuracy
    pass


# TODOs

def calculate_gini(data_partition_torch):
    gini = calculate_gini_impurity(data_partition_torch)
    return gini


def calculate_gini_impurity(partition):
    size = partition.shape[0]
    length = partition.shape[1] - 1
    if size == 0:  # To handle the division by zero
        return torch.tensor(0)
    proportions = torch.Tensor([((partition[:, length] == label).sum() / size)**2 for label in CLASSES])
    return 1 - torch.sum(proportions)


def calculate_total_gini(node_left, node_right):
    size_left = node_left.data_torch_partition.shape[0]
    size_right = node_right.data_torch_partition.shape[0]
    size_total = size_left + size_right
    gini_left = calculate_gini_impurity(node_left.data_torch_partition)
    gini_right = calculate_gini_impurity(node_right.data_torch_partition)
    gini_total = (size_left / size_total) * gini_left + (size_right / size_total) * gini_right
    return gini_total

