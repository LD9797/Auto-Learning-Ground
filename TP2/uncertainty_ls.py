from torchmetrics import PearsonCorrCoef

import codification
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score
import torch
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# One-hot-vector codification is a process in codification.py. The dataset "X" and tags "y" are returned.
# X -> Dataset
# y -> tags
X, y = codification.perform_codification()


# 1.

def split_dataset(data, tags, test_size=0.20, random_state=None):
    """
    Function to partition the dataset.
    Default 80% train and 20% test.
    :returns: x_train, x_test, y_train, y_test
    """
    x_train, x_test, y_train, y_test = train_test_split(data, tags, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


def get_optimum_w_square_means(t, x):
    """
    Function to obtain the optimal w for t (tags) and x (data).
    :returns: optimal w
    """
    m_pinv = torch.pinverse(x)
    w_opt = m_pinv.mm(t)
    return w_opt


def evaluate_model_original(data, w, activation_function=False):
    """
    Function to evaluate the model.
    It consists of the dot product between the data and the optimum w.
    If the result is greater than zero then the class is 1. Otherwise, the class is 0.
    :returns: the estimated tags.
    """
    y_out = data.mm(w)
    if activation_function:
        y_out[y_out >= 0] = 1
        y_out[y_out < 0] = 0
    t_estimated = y_out
    return t_estimated


def evaluate_mean_square_error(t, t_estimated):
    """
    Calculates the mean squared error for the tags and the estimated tags of the model.
    :return: mean squared error
    """
    error = torch.norm(t - t_estimated, 2) / (2 * t.shape[0])
    return error


def run_30():
    """
    Partitions the data, calculates the optimum w, and returns the error, accuracy, and f1 scores.
    It performs this process 30 times.
    :return: a log of the results.
    """
    log = {}
    for i in range(30):
        x_train, x_test, y_train, y_test = split_dataset(X, y)

        # Training model
        train = torch.tensor(x_train)
        targets = torch.tensor(y_train.to_numpy()).unsqueeze(1).to(torch.float64)
        w_opt = get_optimum_w_square_means(targets, train)

        # Testing model
        test = torch.tensor(x_test)
        test_targets = torch.tensor(y_test.to_numpy()).unsqueeze(1).to(torch.float64)
        targets_estimated = evaluate_model_original(test, w_opt, activation_function=True)

        # Calculating error, accuracy and f1 score
        error = evaluate_mean_square_error(test_targets, targets_estimated)
        accuracy = accuracy_score(test_targets, targets_estimated)
        f1s = f1_score(test_targets, targets_estimated, average='weighted')

        # Saving results in the log
        log[i] = {'error': error, 'accuracy': accuracy, 'f1s': f1s}
    return log


# 2. ECE

def calculate_expected_calibration_error(x_in, y_real, uncertainties, y_predicted, n_bins=10, plot=True):
    # Calculate bins ranges
    bins = create_bins_and_append_prediction_to_values(uncertainties, y_real, y_predicted, n_bins)
    # Calculate the accuracy of each bin
    accuracy_bins = torch.tensor([calculate_accuracy_bin(bins[bin_num]) for bin_num in bins])

    # Calculate the average uncertainty of each bin
    bin_average_uncertainty = [sum(data[0] for data in bins[in_bin]) / len(bins[in_bin]) for in_bin in bins]
    bin_average_uncertainty = torch.tensor(bin_average_uncertainty)

    # bin_average_uncertainty -> Acts as X | accuracy_bins -> Acts as Y

    # Pearson coefficient calculation

    # Method 2: calculation with torch metrics
    pearson = PearsonCorrCoef()
    pearson_corr = pearson(bin_average_uncertainty, accuracy_bins).nan_to_num().item()

    # Plotting
    if plot:
        plot_bin_accuracy_chart(bins, accuracy_bins, bin_average_uncertainty)

    ece = 1 - abs(pearson_corr)
    return max(0, min(ece, 1))


def create_bins_and_append_prediction_to_values(uncertainties, y_real, y_predicted, n_bins):
    # Convert uncertainties to numpy array for quantile calculations
    uncertainties_np = torch.tensor(uncertainties).to(torch.float64)

    # Calculate quantiles to define bin edges
    quantiles = torch.linspace(0, 1, n_bins + 1).to(torch.float64)
    bin_edges = torch.quantile(uncertainties_np, quantiles)

    # Create bins
    bins = {i: [] for i in range(n_bins)}

    # Assign values to bins
    for uncertainty, real_label, predicted_label in zip(uncertainties, y_real, y_predicted):
        # Determine the bin index by finding the first bin edge that is greater than the uncertainty
        index = torch.searchsorted(bin_edges, uncertainty, right=True) - 1
        index = min(index.item(), n_bins - 1)  # Ensure the value equal to max_val is included in the last bin
        prediction = (predicted_label >= 0).to(torch.float64) == real_label
        bins[index].append((uncertainty, prediction))

    return bins


def calculate_accuracy_bin(in_bin):
    if len(in_bin) == 0:
        return 0
    correct_predictions = 0
    for value in in_bin:
        correct_predictions += 1 if value[1] else 0
    total_predictions = len(in_bin)
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


def plot_bin_accuracy_chart(bins, accuracy_bins, bin_average_uncertainty):
    bin_edges_visual = torch.linspace(min(bins[0])[0].item(), max(bins[len(bins) - 1])[0].item(), steps=len(bins) + 1)
    bin_centers_visual = 0.5 * (bin_edges_visual[:-1] + bin_edges_visual[1:])

    # Create the figure and axis.
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot a dot in the middle of each bin.
    for center, value in zip(bin_centers_visual, accuracy_bins):
        ax.plot(center, value, 'ro')  # 'ro' for red circle

    # Print average uncertainty for each bin in x-axis.
    custom_ticks = [center for center in bin_centers_visual]
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels([f"#{bin_num + 1}" for bin_num in range(len(bin_average_uncertainty))])

    # Add a line for each bin edge
    for edge in bin_edges_visual:
        ax.axvline(edge, color='red', linestyle='dashed', linewidth=1)

    # Set axis labels and title.
    ax.set_xlabel('')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy per Bin for ECE calculation')

    # Set y-axis limits to fit the range of bin values.
    ax.set_ylim(-10, 110)

    # Print bin number in secondary x-axis.
    ax2 = ax.twiny()
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 17))
    ax2.set_xlabel('Bin Number and Average Uncertainty (Variance)')
    ax2.set_xticks(custom_ticks)
    custom_labels = [f"{"{:.1e}".format(round(bin_num.item(), 5))}" for index, bin_num in
                     enumerate(bin_average_uncertainty)]
    ax2.set_xticklabels(custom_labels)
    ax2.set_xlim(ax.get_xlim())
    ax2.spines['bottom'].set_visible(False)

    plt.show()


def test_calculate_expected_calibration_error_worse_calibration():
    example_variances = torch.tensor([0.000814, 0.000343, 0.000491, 0.000273, 0.000321, 0.000325, 0.000034, 0.000678, 0.000084, 0.00041])
    example_y_outputs = torch.tensor([0.871897, - 0.095776, - 0.199759, 1.157389, 0.4576, 0.455021, 0.95016, 0.559645, 0.901524, 0.493699])
    example_y_real = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    ece = calculate_expected_calibration_error(None, example_y_real, example_variances, example_y_outputs, plot=False)
    assert ece == 1
    print(f"Passed! \nCalculated ECE: {ece} \nExpected: 1")


def test_calculate_expected_calibration_error_perfect_calibration():
    example_variances = torch.tensor([0.000814, 0.000343, 0.000491, 0.000273, 0.000321, 0.000325, 0.000034, 0.000678, 0.000084, 0.00041])
    example_y_outputs = torch.tensor([0.871897, -0.095776, -0.199759, 1.157389, -0.4576, -0.455021, 0.95016, 0.559645, 0.901524, 0.493699])
    example_y_real = [1, 0, 0, 1, 0, 0, 1, 1, 1, 1]
    ece = calculate_expected_calibration_error(None, example_y_real, example_variances, example_y_outputs, plot=False)
    assert ece == 0
    print(f"Passed! \nCalculated ECE: {ece} \nExpected: 0")


# 3.
def quantify_uncertainty_ensemble(x_test, model, n_ensemble=10, ensemble=None, random_state=None):
    if ensemble is None:
        ensemble = train_ensemble(n_ensemble, random_state=random_state)
    y_outputs = run_ensemble_uq(x_test, ensemble, model)
    y_outputs_stacked = torch.stack(y_outputs).squeeze(-1).t()
    variances = []
    predictions = []
    for y_out in y_outputs_stacked:
        var_xi = torch.var(y_out)
        predicted_y = torch.mean(y_out)
        variances.append(var_xi)
        predictions.append(predicted_y)
    return torch.tensor(variances), torch.tensor(predictions)


def train_ensemble(n, x_train=None, y_train=None, random_state=None):
    if x_train is None or y_train is None:
        x_train, _, y_train, _ = split_dataset(X, y, test_size=0.3, random_state=random_state)
    train_splits = kfold_split(x_train, y_train.to_numpy(), n, random_state=random_state)
    ensembles = []
    for split in train_splits:
        x_split = torch.tensor(split[0])
        y_split = torch.tensor(split[1]).unsqueeze(1).to(torch.float64)
        w_opt = get_optimum_w_square_means(y_split, x_split)
        ensembles.append(w_opt)
    return ensembles


def kfold_split(features, labels, n_splits, random_state=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)  # Using a random state for reproducibility
    # This will store the training subsets and their corresponding labels
    subsets = []

    for train_index, test_index in kf.split(features):
        feature_subset = features[test_index]
        label_subset = labels[test_index]
        subsets.append((feature_subset, label_subset))

    return subsets


def run_ensemble_uq(x, ensemble, model):
    y_outputs = [model(x, indv_ensemble) for indv_ensemble in ensemble]
    return y_outputs


def test_quantify_uncertainty_ensemble_individual_entry():
    # Splitting dataset with random state 42 to ensure reproducibility
    x_train, x_test, y_train, y_test = split_dataset(X, y, random_state=42)
    # Individual entry:
    entry = torch.tensor(x_test[0]).unsqueeze(-1).t().to(torch.float64)
    variance, y_outputs = quantify_uncertainty_ensemble(entry, evaluate_model_original, n_ensemble=10, random_state=42)
    variance = round(variance.item(), 3)
    y_outputs = round(y_outputs.item(), 3)
    assert variance == 0.002
    assert y_outputs == 0.011
    print("Single Entry Test:")
    print(f"Result: Passed! \nCalculated Variance: {variance} \nExpected: {0.002}")
    print(f"Estimated Output: {y_outputs} \nExpected: {0.011}")


def test_quantify_uncertainty_ensemble_multiple_entry():
    # Splitting dataset with random state 42 to ensure reproducibility
    x_train, x_test, y_train, y_test = split_dataset(X, y, random_state=42)
    # Gathering entries:
    x_test = torch.tensor(x_test)
    entries = torch.stack([x_test[0], x_test[1], x_test[2]])
    variance, y_outputs = quantify_uncertainty_ensemble(entries, evaluate_model_original, n_ensemble=10, random_state=42)
    variance = [round(x, 4) for x in list(variance.numpy())]
    y_outputs = [round(x, 4) for x in list(y_outputs.numpy())]
    assert variance == [0.0024, 0.0001, 0.0004]
    assert y_outputs == [0.0108, 0.9777, 0.8247]
    print("Multiple Entry Test:")
    print(f"Result: Passed! \nCalculated Variance: {variance} \nExpected: [0.0024, 0.0001, 0.0004]")
    print(f"Estimated Outputs: {y_outputs} \nExpected: [0.0108, 0.9777, 0.8247]")


# 4.
def run_tests(n):
    start_time = time.time()

    x_train, x_test, y_train, y_test = split_dataset(X, y, test_size=0.30, random_state=42)
    test_partitions = kfold_split(x_test, y_test.to_numpy(), n_splits=10, random_state=42)
    train_partitions = kfold_split(x_train, y_train.to_numpy(), n_splits=10, random_state=42)

    trained_ensemble = train_ensemble(n, x_train, y_train, random_state=42)
    test_part_avg_ece, test_part_std = calculate_avg_ece_and_std_partition(test_partitions, trained_ensemble, "Test")
    train_part_avg_ece, train_part_std = calculate_avg_ece_and_std_partition(train_partitions, trained_ensemble,
                                                                             "Train")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Results for configuration N={n}")
    print(f"Elapsed time: {elapsed_time}")
    print(f"Test Partition Average ECE: {test_part_avg_ece}, Test Partition Std: {test_part_std}")
    print(f"Train Partition Average ECE: {train_part_avg_ece}, Test Partition Std: {train_part_std}")


def calculate_avg_ece_and_std_partition(partition, ensemble, partition_label):
    test_partition_ece = []
    for part in partition:
        x_partition = torch.tensor(part[0])
        y_real_partition = torch.tensor(part[1]).unsqueeze(1).to(torch.float64)
        variance, y_outputs = quantify_uncertainty_ensemble(x_partition, evaluate_model_original, ensemble=ensemble)
        ece = calculate_expected_calibration_error(x_partition, y_real_partition, variance, y_outputs, plot=False)
        test_partition_ece.append(ece)
    test_partition_ece = torch.tensor(test_partition_ece)
    average_ece = torch.mean(test_partition_ece)
    std = torch.std(test_partition_ece)
    plot_results(test_partition_ece, partition_label, average_ece, std, len(ensemble))
    return average_ece, std


def plot_results(test_partition_ece, partition_label, average_ece, std, n_configuration):
    x_linspace = torch.linspace(0, len(test_partition_ece), steps=len(test_partition_ece) + 1)
    y_centers = 0.5 * (x_linspace[:-1] + x_linspace[1:])

    # Create the figure and axis.
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot a dot in the middle.
    for center, value in zip(y_centers, test_partition_ece):
        ax.plot(center, value, 'ro')  # 'ro' for red circle

    # Print labels.
    custom_ticks = [center for center in y_centers]
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels([f"#{bin_num + 1}" for bin_num in range(len(test_partition_ece))])

    # Add a line for each partition.
    for edge in x_linspace:
        ax.axvline(edge, color='red', linestyle='dashed', linewidth=1)

    ax.set_xlabel('Partition number')
    ax.set_ylabel('ECE')
    ax.set_title(f'{partition_label} Partition ECE values | Configuration N={n_configuration}'
                 f' \n Average ECE: {round(average_ece.item(), 3)} '
                 f'| Standard Deviation: {round(std.item(), 3)}')

    plt.show()


#run_tests(10)
#run_tests(100)
#run_tests(1000)




