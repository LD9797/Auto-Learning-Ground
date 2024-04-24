from codification import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# X, y -> dataset & tags

def split_dataset(data, tags, test_size=0.20):
    # random_state ensure random split
    x_train, x_test, y_train, y_test = train_test_split(data, tags, test_size=test_size)
    # print(f"X_train = {x_train.shape}")
    # print(f"X_test = {x_test.shape}")
    return x_train, x_test, y_train, y_test


def get_optimum_w_square_means(t, x):
    m_pinv = torch.pinverse(x)
    w_opt = m_pinv.mm(t)
    return w_opt


def estimate_optimum_w_GD(TargetsAll, SamplesAll, alpha, epochs):
    SamplesAlltrans = SamplesAll.transpose(0, 1)
    error_per_epoch = np.zeros(epochs)
    wT = torch.rand((56, 1), dtype=torch.float64)
    print("SamplesAll.shape[0] \n", SamplesAll.shape[0])
    for epoch in range(0, epochs):
        errorGradient = (1 / SamplesAll.shape[0]) * SamplesAlltrans.mm(SamplesAll).mm(wT) - SamplesAlltrans.mm(TargetsAll)
        wT -= alpha * errorGradient
        EstimatedTargetsAll, _ = evaluate_model(SamplesAll, wT)
        error = evaluate_mean_square_error(TargetsAll, EstimatedTargetsAll)
        error_per_epoch[epoch] = error
    return wT, error_per_epoch


def evaluate_mean_square_error(t, t_estimated):
    error = torch.norm(t - t_estimated, 2) / (2 * t.shape[0])
    return error


def evaluate_model_original(data, w):
    y_out = data.mm(w)
    y_out[y_out >= 0] = 1
    y_out[y_out < 0] = 0
    t_estimated = y_out
    return t_estimated


def evaluate_model(data, w):
    y_out = data.mm(w)
    probabilities = F.sigmoid(y_out)  # Sigmoid to transform y_out to probabilities
    t_estimated = (probabilities >= 0.5).float()  # Threshold probabilities to get binary labels
    # Returns exact same t_estimated as 'evaluate_model_original'
    return t_estimated, probabilities


def run_30():
    log = {}
    log_gd = {}
    for i in range(30):
        x_train, x_test, y_train, y_test = split_dataset(X, y)

        # Train
        train = torch.tensor(x_train)
        targets = torch.tensor(y_train.to_numpy()).unsqueeze(1).to(torch.float64)
        w_opt = get_optimum_w_square_means(targets, train)
        # Test
        test = torch.tensor(x_test)
        test_targets = torch.tensor(y_test.to_numpy()).unsqueeze(1).to(torch.float64)
        targets_estimated, probabilities = evaluate_model(test, w_opt)

        error = evaluate_mean_square_error(test_targets, targets_estimated)
        accuracy = accuracy_score(test_targets, targets_estimated)
        f1s = f1_score(test_targets, targets_estimated, average='weighted')
        ece = calculate_expected_calibration_error(probabilities, test_targets, 10)

        log[i] = {'error': error, 'accuracy': accuracy, 'f1s': f1s, 'ece': ece}

        #w_opt_gd, _ = estimate_optimum_w_GD(targets, train, 0.001, 5)
        #targets_estimated_gd, probabilities_gd = evaluate_model(test, w_opt_gd)
        #error_gd = evaluate_mean_square_error(test_targets, targets_estimated_gd)
        #accuracy_gd = accuracy_score(test_targets, targets_estimated_gd)
        #f1s_gd = f1_score(test_targets, targets_estimated_gd, average='weighted')
        #ece_gd = calculate_expected_calibration_error(probabilities_gd, test_targets, 10)

        #log_gd[i] = {'error': error_gd, 'accuracy': accuracy_gd, 'f1s': f1s_gd, 'ece': ece_gd}

    return log


def calculate_expected_calibration_error(probabilities, true_labels, n_bins=10):
    bin_limits = torch.linspace(0, 1, steps=n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        # Checking for probabilities in between the bin limit.
        bin_mask = (probabilities >= bin_limits[i]) & (probabilities < bin_limits[i+1])
        if bin_mask.any():
            # Getting probabilities using the mask.
            bin_probabilities = probabilities[bin_mask]
            # Getting the true labels for the selected probabilities using the mask.
            bin_labels = true_labels[bin_mask]
            # Todas las predicciones correctas dividido entre el tamaño del bin (promedio)
            bin_accuracy = (bin_labels == (bin_probabilities >= 0.5).float()).float().mean()
            # La suma de todas las probabilidades divido entre el tamaño del bin (promedio)
            bin_confidence = bin_probabilities.mean()
            # Tasa de aciertos promedio - promedio del puntaje de incertidumbre
            bin_error = torch.abs(bin_accuracy - bin_confidence)
            # Numero de predicciones en el bin dividido entre el total, multiplicado por el error del bin
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


def quantify_uncertainty_ensemble(x, model, n=10):
    ensembles = create_ensembles(n)
    y_outputs = torch.tensor([run_ensemble_uq(x, ensemble) for ensemble in ensembles])
    variance = 0
    for y_output in y_outputs:
        variance += (y_outputs.mean() - y_output) ** 2
    variance = variance/(n-1)
    return variance


def train_ensemble(x_train, y_train):
    train = torch.tensor(x_train)
    targets = torch.tensor(y_train.to_numpy()).unsqueeze(1).to(torch.float64)
    w_opt = get_optimum_w_square_means(targets, train)
    return w_opt


def run_ensemble_uq(x, ensemble):
    y_out = x.unsqueeze(-1).t().mm(ensemble)
    return y_out


def create_ensembles(n=10):
    ensembles = []
    for i in range(n):
        x_train, _, y_train, _ = split_dataset(X, y)
        ensemble = train_ensemble(x_train, y_train)
        ensembles.append(ensemble)
    return ensembles


def quantify_test():
    x_train, _, y_train, _ = split_dataset(X, y)
    x_in = torch.tensor(x_train[0]).to(torch.float64)
    variance = quantify_uncertainty_ensemble(x_in, _, n=10)
    print(variance)


if __name__ == '__main__':
    quantify_test()
