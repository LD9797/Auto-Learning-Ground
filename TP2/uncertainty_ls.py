import codification
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import warnings
from torchmetrics.regression import PearsonCorrCoef
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# One-hot-vector codification is a process in codification.py. The dataset "X" and tags "y" are returned.
# X -> Dataset
# y -> tags
X, y = codification.perform_codification()


# 1.

def split_dataset(data, tags, test_size=0.20):
    """
    Function to partition the dataset.
    Default 80% train and 20% test.
    :returns: x_train, x_test, y_train, y_test
    """
    x_train, x_test, y_train, y_test = train_test_split(data, tags, test_size=test_size)
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

def calculate_expected_calibration_error_alt(x_in, y_real, uncertainties, y_predicted, n_bins=10):
    # Calculate bins ranges
    bins = create_bins_and_append_prediction_to_values(uncertainties, y_real, y_predicted, n_bins)
    # Calculate the accuracy of each bin
    accuracy_bins = torch.tensor([calculate_accuracy_bin(bins[bin_num]) for bin_num in bins])
    # Calculate the average uncertainty of each bin
    bin_average_uncertainty = [sum(data[0] for data in bins[in_bin]) / len(bins[in_bin]) for in_bin in bins]
    bin_average_uncertainty = torch.tensor(bin_average_uncertainty)

    # bin_average_uncertainty -> Acts as X
    # accuracy_bins -> Acts as Y

    # Pearson coefficient calculation

    # Method 1: correlation matrix
    corr_matrix = torch.corrcoef(torch.stack((bin_average_uncertainty, accuracy_bins)))
    pearson_corr = corr_matrix[0, 1]

    # Method 2: calculation with torch metrics
    pearson = PearsonCorrCoef()
    coeff = pearson(bin_average_uncertainty, accuracy_bins)

    # Linear Regression model - Line that describes the points
    model = LinearRegression()
    model.fit(bin_average_uncertainty.unsqueeze(-1), accuracy_bins.unsqueeze(-1))

    # Calculate distances from the points to the line
    m = torch.tensor(model.coef_[0])
    c = torch.tensor(model.intercept_)
    distances = torch.abs(m * bin_average_uncertainty.flatten() - accuracy_bins + c) / torch.sqrt(m ** 2 + 1)

    # Calculate mean distance
    mean_distance_ece = torch.mean(distances)

    print(f"Pearson coefficient: {coeff}")
    print(f"Mean Distance ECE: {mean_distance_ece}")

    # Plotting

    bin_edges_visual = torch.linspace(min(bins[0])[0].item(), max(bins[len(bins) - 1])[0].item(), steps=len(bins) + 1)
    bin_centers_visual = 0.5 * (bin_edges_visual[:-1] + bin_edges_visual[1:])

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot a dot in the middle of each bin
    for center, value in zip(bin_centers_visual, accuracy_bins):
        ax.plot(center, value, 'ro')  # 'ro' for red circle

    # Print average uncertainty for each bin
    custom_ticks = [center for center in bin_centers_visual]
    ax.set_xticks(custom_ticks)  # Set the positions for the ticks
    ax.set_xticklabels([f"#{bin_num + 1}" for bin_num in range(len(bin_average_uncertainty))])  # Set the custom labels for the ticks

    # Add a line for each bin edge
    for edge in bin_edges_visual:
        ax.axvline(edge, color='red', linestyle='dashed', linewidth=1)

    # Set axis labels and title
    ax.set_xlabel('')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy per Bin')

    # Set y-axis limits to fit the range of bin values
    ax.set_ylim(-10, 110)

    # Print bin number in secondary axis.
    ax2 = ax.twiny()
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 17))
    ax2.set_xlabel('Bin Number and Average Uncertainty (Variance)')
    ax2.set_xticks(custom_ticks)
    custom_labels = [f"{"{:.1e}".format(round(bin_num.item(), 5))}" for index, bin_num in enumerate(bin_average_uncertainty)]
    ax2.set_xticklabels(custom_labels)
    ax2.set_xlim(ax.get_xlim())
    ax2.spines['bottom'].set_visible(False)
    # ax.set_xlim(min(bin_edges), max(bin_edges))

    # Predictions from the model
    # y_pred = model.predict(bin_centers.unsqueeze(-1))

    # Distance line
    # for i in range(len(bin_centers)):
    # ax.plot([bin_centers[i].item(), bin_centers[i].item()], [accuracy_bins[i].item(), y_pred[i].item()], 'g-')

    # Plotting linear regression line
    # ax.plot(torch.linspace(min(bin_centers).item(), max(bin_centers).item(), steps=len(bins)), y_pred, color='blue')

    # Show plot
    plt.show()


def calculate_accuracy_bin(in_bin):
    if len(in_bin) == 0:
        return 0
    correct_predictions = 0
    for value in in_bin:
        correct_predictions += 1 if value[1] else 0
    total_predictions = len(in_bin)
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


def create_bins_and_append_prediction_to_values(uncertainties, y_real, y_predicted, n_bins):
    # Convert uncertainties to numpy array for quantile calculations
    uncertainties_np = np.array(uncertainties)

    # Calculate quantiles to define bin edges
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(uncertainties_np, quantiles)

    # Create bins
    bins = {i: [] for i in range(n_bins)}

    # Assign values to bins
    for uncertainty, real_label, predicted_label in zip(uncertainties, y_real, y_predicted):
        # Determine the bin index by finding the first bin edge that is greater than the uncertainty
        index = np.searchsorted(bin_edges, uncertainty, side='right') - 1
        index = min(index, n_bins - 1)  # Ensure the value equal to max_val is included in the last bin
        prediction = (predicted_label >= 0).to(torch.float64) == real_label
        bins[index].append((uncertainty, prediction))

    return bins


# 3.
def quantify_uncertainty_ensemble(x, model, n=10):
    ensembles = train_ensemble(n)
    y_outputs = [run_ensemble_uq(x, ensemble, model) for ensemble in ensembles]
    y_outputs_stacked = torch.stack(y_outputs).squeeze(-1).t()
    variances = []
    predictions = []
    for y_out in y_outputs_stacked:
        var_xi = torch.var(y_out)
        predicted_y = torch.mean(y_out)
        variances.append(var_xi)
        predictions.append(predicted_y)
    return torch.tensor(variances), torch.tensor(predictions)


def train_ensemble(n):
    x_train, _, y_train, _ = split_dataset(X, y, test_size=0.3)
    train_splits = kfold_split(x_train, y_train.to_numpy(), n)
    ensembles = []
    for split in train_splits:
        x_split = torch.tensor(split[0])
        y_split = torch.tensor(split[1]).unsqueeze(1).to(torch.float64)
        w_opt = get_optimum_w_square_means(y_split, x_split)
        ensembles.append(w_opt)
    return ensembles


def kfold_split(features, labels, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # Using a random state for reproducibility
    # This will store the training subsets and their corresponding labels
    subsets = []

    for train_index, test_index in kf.split(features):
        feature_subset = features[test_index]
        label_subset = labels[test_index]
        subsets.append((feature_subset, label_subset))

    return subsets


def run_ensemble_uq(x, ensemble, model):
    y_out = model(x, ensemble)
    return y_out


def quantify_test():
    x_train, x_test, y_train, y_test = split_dataset(X, y)
    # Individual entry: x_in = torch.tensor(x_test[0]).unsqueeze(-1).t().to(torch.float64)
    x_test = torch.tensor(x_test)
    variance, y_outputs = quantify_uncertainty_ensemble(x_test, evaluate_model_original, n=10)
    calculate_expected_calibration_error_alt(x_test, y_test, variance, y_outputs)


quantify_test()

# 4. TODO Pending


