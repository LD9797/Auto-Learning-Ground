from codification import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5, stratify=y)
print(f"X_train = {X_train.shape}")
print(f"X_test = {X_test.shape}")

# Mi Modelo


def get_optimum_w_square_means(t, model):
    m_pinv = torch.pinverse(model)  # torch.tensor(np.linalg.pinv(M.numpy()))
    w_opt = m_pinv.mm(t)
    return w_opt


def evaluate_mean_square_error(t, t_estimated):
    error = torch.norm(t - t_estimated, 2) / (2 * t.shape[0])
    return error


def evaluate_model(model, w):
    y = model.mm(w)
    y[y > 0] = 1
    y[y <= 0] = 0
    t_estimated = y
    return t_estimated


# Train
train = torch.tensor(X_train)
targets = torch.tensor(y_train.to_numpy()).unsqueeze(1).to(torch.float64)
w_opt = get_optimum_w_square_means(targets, train)

# Test
test = X_test# [:, :-1]
test = torch.tensor(test)
test_targets = torch.tensor(y_test.to_numpy()).unsqueeze(1).to(torch.float64)
targets_estimated = evaluate_model(test, w_opt)

error = evaluate_mean_square_error(test_targets, targets_estimated)
print(error)

accuracy = accuracy_score(test_targets, targets_estimated)

# accuracy_score(test_targets, targets_estimated)
#accuracy = accuracy_score(y_test, y_predictions)
#recall = recall_score(y_test, y_predictions, average='weighted')
#precision = precision_score(y_test, y_predictions, average='weighted')
#f1s = f1_score(y_test, y_predictions, average='weighted')


