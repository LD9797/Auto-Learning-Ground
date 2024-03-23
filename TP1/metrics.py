# Lecture 7 example
import torch

matrix = [
    [51, 2, 0],
    [30, 31, 2],
    [0, 10, 11]
]


def true_positives(the_class):
    return matrix[the_class][the_class]


def true_negatives(the_class):
    predictions = 0
    for index_row, row in enumerate(matrix):
        if index_row != the_class:
            predictions += sum(element for index_element, element in enumerate(row) if index_element != the_class)
    return predictions


def false_negatives(the_class):
    predictions = sum(row[the_class] for index, row in enumerate(matrix) if index != the_class)
    return predictions


def false_positives(the_class):
    class_row = matrix[the_class]
    predictions = sum(value for index, value in enumerate(class_row) if index != the_class)
    return predictions


def calculate_metrics(the_class):
    the_class = the_class - 1
    tp = true_positives(the_class)
    tn = true_negatives(the_class)
    fp = false_positives(the_class)
    fn = false_negatives(the_class)
    return tp, tn, fp, fn


my_class = 3
tp, tn, fp, fn = calculate_metrics(my_class)

# Class 3
print(f"TP={tp}")  # 11
print(f"TN={tn}")  # 114
print(f"FP={fp}")  # 10
print(f"FN={fn}")  # 2


def get_stats(matrix, c):
    tp = matrix[c, c]
    fn = torch.sum(matrix[:, c]) - tp
    fp = torch.sum(matrix[c, :]) - tp
    tn = torch.sum(matrix) - tp - fp - fn
    return tp, tn, fp, fn


print(get_stats(torch.Tensor(matrix), 2))

