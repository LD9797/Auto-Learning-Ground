def create_bins_and_append_prediction_to_values_old(uncertainties, y_true, n_bins):
    # Define range and compute bin width
    min_val, max_val = min(uncertainties), max(uncertainties)
    bin_width = (max_val - min_val) / n_bins

    # Create bins
    bins = {i: [] for i in range(n_bins)}

    # Assign values to bins
    for value, true_label in zip(uncertainties, y_true):
        index = int((value - min_val) / bin_width)
        index = min(index, n_bins - 1)  # Ensure the value equal to max_val is included in the last bin
        prediction = (value >= 0).to(torch.float64) == true_label
        bins[index].append((value, prediction))

    return bins


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

def create_bins_and_append_prediction_to_values_old(uncertainties, y_true, y_predicted, n_bins):
    # Define range and compute bin width
    min_val, max_val = min(uncertainties), max(uncertainties)
    bin_width = (max_val - min_val) / n_bins

    # Create bins
    bins = {i: [] for i in range(n_bins)}

    # Assign values to bins
    for uncertainty, true_label, predicted_label in zip(uncertainties, y_true, y_predicted):
        index = int((uncertainty - min_val) / bin_width)
        index = min(index, n_bins - 1)  # Ensure the value equal to max_val is included in the last bin
        prediction = (predicted_label >= 0).to(torch.float64) == true_label
        bins[index].append((uncertainty, prediction))

    return bins


def run_ece():
    # Train model
    x_train, x_test, y_train, y_test = split_dataset(X, y)
    train = torch.tensor(x_train)
    targets = torch.tensor(y_train.to_numpy()).unsqueeze(1).to(torch.float64)
    w_opt = get_optimum_w_square_means(targets, train)

    # Testing model
    test = torch.tensor(x_test)
    test_targets = torch.tensor(y_test.to_numpy()).unsqueeze(1).to(torch.float64)
    targets_estimated = evaluate_model_original(test, w_opt)

    calculate_expected_calibration_error_alt(test, test_targets, targets_estimated)


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
