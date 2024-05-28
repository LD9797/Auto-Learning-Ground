import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

"""
This is the formatted code for the one-hot-vector encoding of the data. 
Only necessary code is included. 
"""

# Read data frame
df = pd.read_csv('UNSW_NB15_training-set.csv')


def basic_pre_processing():
    list_drop = ['id', 'attack_cat']
    df.drop(list_drop, axis=1, inplace=True)


def codification_cat_data():
    X = df.iloc[:, :-1]
    X.head()
    df_cat = df.select_dtypes(exclude=[np.number])
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
            df[feature] = np.where(df[feature].isin(df[feature].value_counts().head().index), df[feature], '-')


def perform_codification():
    basic_pre_processing()
    codification_cat_data()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 2, 3])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    return X, y
