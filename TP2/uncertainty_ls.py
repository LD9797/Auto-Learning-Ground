import codification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn.functional as F
import numpy as np
import warnings
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


def evaluate_model_original(data, w):
    """
    Function to evaluate the model.
    It consists of the dot product between the data and the optimum w.
    If the result is greater than zero then the class is 1. Otherwise, the class is 0.
    :returns: the estimated tags.
    """
    y_out = data.mm(w)
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
        targets_estimated = evaluate_model_original(test, w_opt)

        # Calculating error, accuracy and f1 score
        error = evaluate_mean_square_error(test_targets, targets_estimated)
        accuracy = accuracy_score(test_targets, targets_estimated)
        f1s = f1_score(test_targets, targets_estimated, average='weighted')

        # Saving results in the log
        log[i] = {'error': error, 'accuracy': accuracy, 'f1s': f1s}
    return log


# 2. TODO Pending
# Sigmoid implementation below.

def evaluate_model_sigmoid(data, w):
    y_out = data.mm(w)
    probabilities = F.sigmoid(y_out)  # Sigmoid to transform y_out to probabilities
    t_estimated = (probabilities >= 0.5).float()  # Threshold probabilities to get binary labels
    # Returns exact same t_estimated as 'evaluate_model_original'
    return t_estimated, probabilities


def calculate_expected_calibration_error(probabilities, true_labels, n_bins=10):
    bin_limits = torch.linspace(0, 1, steps=n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (probabilities >= bin_limits[i]) & (probabilities < bin_limits[i+1])
        if bin_mask.any():
            bin_probabilities = probabilities[bin_mask]
            bin_labels = true_labels[bin_mask]
            bin_accuracy = (bin_labels == (bin_probabilities >= 0.5).float()).float().mean()
            bin_confidence = bin_probabilities.mean()
            bin_error = torch.abs(bin_accuracy - bin_confidence)
            ece += (bin_mask.float().mean() * bin_error)
    return ece.item()


def test_calculate_expected_calibration_error_10_bins():
    samples = torch.tensor([[0.78], [0.36], [0.08], [0.58], [0.49], [0.85], [0.30], [0.63], [0.17]])
    true_labels = torch.tensor([[0], [1], [0], [0], [0], [0], [1], [1], [1]])
    ece = calculate_expected_calibration_error(samples, true_labels, n_bins=10)
    assert round(ece, 3) == 0.538


def test_calculate_expected_calibration_error_5_bins():
    samples = torch.tensor([[0.78], [0.36], [0.08], [0.58], [0.49], [0.85], [0.30], [0.63], [0.17]])
    true_labels = torch.tensor([[0], [1], [0], [0], [0], [0], [1], [1], [1]])
    ece = calculate_expected_calibration_error(samples, true_labels, n_bins=5)
    assert round(ece, 3) == 0.304


# 3. TODO Pending

def quantify_uncertainty_ensemble(x, model, n=10):
    ensembles = [train_ensemble() for i in range(n)]
    y_outputs = torch.tensor([run_ensemble_uq(x, ensemble, model) for ensemble in ensembles])
    # TODO: replace for torch.var(y_outputs)
    variance = 0
    for y_output in y_outputs:
        variance += (y_outputs.mean() - y_output) ** 2
    variance = variance/(n-1)
    return variance, y_outputs


def train_ensemble():
    x_train, _, y_train, _ = split_dataset(X, y)
    train = torch.tensor(x_train)
    targets = torch.tensor(y_train.to_numpy()).unsqueeze(1).to(torch.float64)
    w_opt = get_optimum_w_square_means(targets, train)
    return w_opt


def run_ensemble_uq(x, ensemble, model):
    y_out = model(x.unsqueeze(-1).t(), ensemble)
    return y_out


# 4. TODO Pending
