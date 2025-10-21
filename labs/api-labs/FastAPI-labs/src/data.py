import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

def split_data(X, y):
    # Different test size & random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, shuffle=True
    )
    return X_train, X_test, y_train, y_test
