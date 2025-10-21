import joblib

def predict_data(X):
    """
    Load the trained RandomForest model and predict.
    """
    model = joblib.load("../model/iris_model.pkl")
    y_pred = model.predict(X)
    return y_pred
