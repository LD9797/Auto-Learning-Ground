import torch
import pandas
import numpy as np
from gini_functions import calculate_gini, calculate_entropy, calculate_total_gini, calculate_total_entropy


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

    def __init__(self, gini_entropy_total_function=calculate_total_entropy,
                 gini_entropy_function="GINI",
                 num_classes=4, num_features=7, ref_cart=None, current_depth=0):
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
        self.num_classes = num_classes
        self.num_features = num_features
        self.current_depth = current_depth  # Profundidad
        self.gini_entropy_total_function = gini_entropy_total_function
        self.leaf = False
        self.gini_function = gini_entropy_function

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
        if self.node_left:
            str_left = self.node_left.to_xml(current_str)
            str_node += str_left
        if self.node_right:
            str_right = self.node_right.to_xml(current_str)
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
        return self.leaf

    def create_with_children(self, current_depth=0, list_selected_features=None, min_gini=0.000001, max_cart_depth=5,
                             min_observations=2):
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

        min_thresh, min_feature, min_gini_thresh = (
            self.select_best_feature_and_thresh(data_torch=self.data_torch_partition,
                                                list_features_selected=list_selected_features))

        self.feature_num = min_feature
        self.threshold_value = min_thresh
        self.gini = min_gini_thresh
        self.gini_entropy_total_function = min_gini_thresh
        self.current_depth = current_depth
        list_selected_features.append(self.feature_num)

        if (min_gini_thresh <= min_gini or
                len(list_selected_features) == self.num_features or
                current_depth == max_cart_depth or
                self.data_torch_partition.shape[0] <= min_observations):
            # This is a leaf
            self.leaf = True
            length = self.data_torch_partition.shape[1] - 1
            tag_values = torch.unique(self.data_torch_partition[:, length:length + 1].squeeze())
            tags, counts = tag_values.unique(return_counts=True)
            most_common_value = tags[counts.argmax()].item()
            self.dominant_class = most_common_value
            # self.accuracy_dominant_class = 1
            return list_selected_features

        left_idx = self.data_torch_partition[:, self.feature_num] < self.threshold_value
        right_idx = self.data_torch_partition[:, self.feature_num] >= self.threshold_value

        dataset_partition_left = self.data_torch_partition[left_idx]
        dataset_partition_right = self.data_torch_partition[right_idx]

        left_child = NodeCart(current_depth=current_depth)
        left_child.data_torch_partition = dataset_partition_left
        left_child.ref_CART = self

        right_child = NodeCart(current_depth=current_depth)
        right_child.data_torch_partition = dataset_partition_right
        right_child.ref_CART = self

        current_depth += 1

        self.node_left = left_child
        self.node_right = right_child

        unique_features_left = list_selected_features.copy()
        unique_features_right = list_selected_features.copy()

        self.node_left.create_with_children(current_depth, unique_features_left,
                                            max_cart_depth=max_cart_depth,
                                            min_gini=min_gini,
                                            min_observations=min_observations)
        self.node_right.create_with_children(current_depth, unique_features_right,
                                             min_gini=min_gini,
                                             max_cart_depth=max_cart_depth,
                                             min_observations=min_observations)

        #list_selected_features.extend(list_selected_features_left)
        #list_selected_features.extend(list_selected_features_right)

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
        def evaluate_threshold(data, feature_num, gini_entropy_total_function):
            root_node = NodeCart(calculate_total_gini)
            root_node.data_torch_partition = data
            root_node.feature_num = feature_num
            threshold_values = torch.unique(data[:, feature_num:feature_num + 1].squeeze())
            value_gini = {}
            for value in threshold_values:
                root_node.threshold_value = value
                left_idx = data[:, root_node.feature_num] < root_node.threshold_value
                right_idx = data[:, root_node.feature_num] >= root_node.threshold_value
                dataset_partition_left = data[left_idx]
                dataset_partition_right = data[right_idx]
                left_child = NodeCart()
                left_child.data_torch_partition = dataset_partition_left
                right_child = NodeCart()
                right_child.data_torch_partition = dataset_partition_right
                gini = gini_entropy_total_function(left_child, right_child)
                value_gini[value] = gini
            minimum_gini = min(value_gini, key=value_gini.get)
            return {minimum_gini.item(): value_gini[minimum_gini].item()}

        if list_features_selected is None:
            list_features_selected = []
        num_features = data_torch.shape[1] - 1
        features_gini = {}
        for feature in range(num_features):
            if feature not in list_features_selected:
                features_gini[feature] = evaluate_threshold(data_torch, feature, self.calculate_total_gini_entropy)
        min_key, min_inner_dict = min(features_gini.items(), key=lambda item: next(iter(item[1].values())))
        result = features_gini[min_key]
        min_feature = min_key
        min_thresh = list(result.keys())[0]
        min_gini = result[min_thresh]
        return min_thresh, min_feature, min_gini

    def calculate_gini(self, data_partition_torch, num_classes=4):
        """
        Calculates the Gini coefficient for a given partition with the given number of classes
        param data_partition_torch: current dataset partition as a tensor
        param num_classes: K number of classes to discriminate from
        returns the calculated Gini coefficient
        """
        def calculate_gini_impurity(partition):
            size = partition.shape[0]
            if size == 0:  # To handle the division by zero
                return torch.tensor(0)
            length = partition.shape[1] - 1
            _, counts = partition[:, length].unique(return_counts=True)
            gini = 1 - torch.sum((counts / size) ** 2)
            return gini
        return calculate_gini_impurity(data_partition_torch)

    def calculate_entropy(self, data_partition_torch, num_classes=4):
        """
        Calculates the Gini coefficient for a given partition with the given number of classes
        param data_partition_torch: current dataset partition as a tensor
        param num_classes: K number of classes to discriminate from
        returns the calculated Gini coefficient
        """
        def calculate_entropy_disorder(partition):
            size = partition.shape[0]
            if size == 0:  # To handle the division by zero
                return torch.tensor(0)
            length = partition.shape[1] - 1
            epsilon = torch.tensor(0.0001)  # Small epsilon to prevent probabilities equal to 0
            _, counts = partition[:, length].unique(return_counts=True)
            probabilities = (counts / size) + epsilon
            entropy = - torch.sum(probabilities * torch.log(probabilities))
            return entropy
        return calculate_entropy_disorder(data_partition_torch)

    def calculate_total_gini_entropy(self, node_left, node_right):
        selected_function = self.calculate_gini if self.gini_function == "GINI" else self.calculate_entropy
        size_left = node_left.data_torch_partition.shape[0]
        size_right = node_right.data_torch_partition.shape[0]
        size_total = size_left + size_right
        gini_entropy_left = selected_function(node_left.data_torch_partition)
        gini_entropy_right = selected_function(node_right.data_torch_partition)
        gini_entropy_total = (size_left / size_total) * gini_entropy_left + (size_right / size_total) * gini_entropy_right
        return gini_entropy_total

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


#dataset = read_dataset(csv_name="wifi_localization_reduced.txt")
#node_cart = NodeCart()
#node_cart.data_torch_partition = dataset
#node_cart.gini_entropy_total_function = calculate_total_gini
#node_cart.select_best_feature_and_thresh(data_torch=node_cart.data_torch_partition)


class CART:  # Este es el arbol
    def __init__(self, dataset_torch, max_cart_depth, min_observations=2):
        """
        CART has only one root node
        """
        # min observations per node
        self.min_observations = min_observations  # Para evitar sobre ajuste, se definen algunos valores para permitir controlar la complejidad
        # Si el arbol se hace muy complejo puede llegar a ser uno que se sobreajusta a los datos.
        self.root = NodeCart(calculate_total_entropy, num_classes=4, num_features=7, ref_cart=self)
        self.root.data_torch_partition = dataset_torch
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

    def build_cart(self):
        """
        Build CART from root
        """
        self.list_selected_features = self.root.create_with_children(max_cart_depth=self.max_CART_depth,
                                                                     min_observations=self.min_observations)

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


def train_cart(dataset_torch, name_xml="", max_cart_depth=5, min_obs_per_leaf=2):
    """
    Train CART model
    """
    tree = CART(dataset_torch=dataset_torch, max_cart_depth=max_cart_depth, min_observations=min_obs_per_leaf)
    tree.build_cart()
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


dataset = read_dataset(csv_name="wifi_localization_reduced.txt")
train_cart(dataset, name_xml="cart.xml")
