import pandas
import numpy as np
import torch
import sklearn.model_selection
import optuna
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics


def read_dataset(csv_name='wifi_localization.txt'):
    data_frame = pandas.read_table(csv_name, sep=r'\s+', names=('A', 'B', 'C', 'D', 'E', 'F', 'G', 'ROOM'),
                                   dtype={'A': np.int64, 'B': np.float64, 'C': np.float64, 'D': np.float64,
                                          'E': np.float64, 'F': np.float64, 'G': np.float64, 'ROOM': np.float64})
    X = data_frame.drop('ROOM', axis=1).values
    y = data_frame['ROOM'].values
    return X, y


def objective(trial):
    # Load and split the dataset
    X, y = read_dataset('wifi_localization.txt')
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=42)

    # Suggest a value for max_depth
    max_depth = trial.suggest_int('max_depth', 1, 32)

    # Create and fit the model
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    preds = clf.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, preds)

    return accuracy * 100  # scale to percentage


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best parameters: {study.best_params}")
print(f"Best accuracy: {study.best_value}%")
