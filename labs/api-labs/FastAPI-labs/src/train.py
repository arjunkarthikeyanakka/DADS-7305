from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
from data import load_data, split_data


def fit_model(X_train, X_test, y_train, y_test):
    """
    Train a Random Forest model with feature scaling and save it.
    """
    # Build a pipeline with scaling + model
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ))
    ])

    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained successfully. Test Accuracy: {acc:.4f}")

    # Save the model
    joblib.dump(clf, "model/iris_model.pkl")


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, X_test, y_train, y_test)
