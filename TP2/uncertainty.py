from codification import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay  # Will plot the confusion matrix
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5, stratify=y)
print(f"X_train = {X_train.shape}")
print(f"X_test = {X_test.shape}")

model_performance = pd.DataFrame(columns=['Accuracy', 'Recall', 'Precision', 'F1-Score', 'time to train',
                                          'time to predict', 'total time'])

# Logistic Regression

start = time.time()
model = LogisticRegression(max_iter=10000, tol=0.0001).fit(X_train, y_train)
end_train = time.time()
y_predictions = model.predict(X_test)
print("y_predictions \n ", y_predictions)
end_predict = time.time()


# Test Model

#y_predictions = np.round(y_predictions).astype(int)
#y_test = np.round(y_test).astype(int)
#print(y_predictions)
#print(y_test)

accuracy = accuracy_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions, average='weighted')
precision = precision_score(y_test, y_predictions, average='weighted')
f1s = f1_score(y_test, y_predictions, average='weighted')


print("Accuracy: " + "{:.2%}".format(accuracy))
print("Recall: " + "{:.2%}".format(recall))
print("Precision: " + "{:.2%}".format(precision))
print("F1-Score: " + "{:.2%}".format(f1s))
print("time to train: " + "{:.2f}".format(end_train-start)+" s")
print("time to predict: " + "{:.2f}".format(end_predict-end_train)+" s")
print("total: " + "{:.2f}".format(end_predict-start)+" s")

