import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# Read data frame
df = pd.read_csv('TP2/UNSW_NB15_training-set.csv')


# Explore the dataset
# Proto / Service / State -> Son los atributos categoricos
def explore_dataset():
    print(f"Shape of data frame: {df.shape[0]}")
    print("Set of 10 records:")
    print(df.head(10))
    print("Description:")
    print(df.describe(include="all"))


# Basic Pre-processing
def basic_pre_processing():
    # Drop the id and attack category, not used
    list_drop = ['id', 'attack_cat']
    df.drop(list_drop, axis=1, inplace=True)
    # print("We need to be careful with the range anda values of the histogram:")
    # Esto es para ver el "Desbalanceo" segun el profe.
    df.hist(column="label")
    # plt.show()
    # Get set of records with label 0 and 1
    # Extract rows with label equal to 0
    rows_with_label_0 = df[df['label'] == 0]
    # print(rows_with_label_0.shape)
    # Extract rows with label equal to 1
    rows_with_label_1 = df[df['label'] == 1]
    # print(rows_with_label_1.shape)


def codification_cat_data():
    # Codificacion de los datos categoricos
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X.head()
    feature_names = list(X.columns)
    # print("Original dimensionality before one hot encoding: ")
    # print("X  shape ", X.shape)

    # print("Number of different values per column using the dataframe ")
    # print(df.describe(include='all'))

    # Data Frame with only the categorical variables
    df_cat = df.select_dtypes(exclude=[np.number])
    #  print(df_cat.describe(include='all'))
    # Service also defines the '-' category (to represent an unknown service)
    # Los valores fuera de los 6 mas frecuentes
    DEBUG = 0
    for feature in df_cat.columns:
        if DEBUG == 1:
            print(feature)
            print('nunique = ' + str(df_cat[feature].nunique()))
            print("Is the cardinality higher than 6? ", df_cat[feature].nunique() > 6)
            print("Number of preserved records: ", sum(df[feature].isin(df[feature].value_counts().head().index)))
            print("New categories: (- takes the rest of categories)", df[feature].value_counts().head().index)
            print('----------------------------------------------------')
        if df_cat[feature].nunique() > 6:
            # preserve only the records with values within the first 5 most frequent values (default by pandas)
            # replace with '-' if the value is not between the 5 most frequent
            # Los valores en los menos frecuentes se pone un guion
            df[feature] = np.where(df[feature].isin(df[feature].value_counts().head().index), df[feature], '-')


basic_pre_processing()
codification_cat_data()
df_cat = df.select_dtypes(exclude=[np.number])
# print(df_cat.describe(include='all'))


X = df.iloc[:, :-1]  # Selecciona todo menos la ultima columna que se llama "label"
y = df.iloc[:, -1]  # Selecciona solo la ultima columna
feature_names = list(X.columns)
# print("Number of features before one hot encoding: ", len(feature_names))
# Create the one hot encoder transformer and transform:
# Columns 1, 2 and 3 are the ones to encode
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 2, 3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# print("Number of features after one hot encoding: ", X.shape)






