import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
from datetime import datetime

def train_and_evaluate():
    # Get configuration from environment variables
    model_path = os.environ.get('MODEL_PATH', '/models')
    n_estimators = int(os.environ.get('N_ESTIMATORS', 100))
    test_size = float(os.environ.get('TEST_SIZE', 0.2))
    
    print(f"Starting model training with config:")
    print(f"  - n_estimators: {n_estimators}")
    print(f"  - test_size: {test_size}")
    print(f"  - model_path: {model_path}")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Cross-validation for robust evaluation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Evaluate on test set
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    
    # Print evaluation metrics
    print(f"\n{'='*50}")
    print("MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Test set accuracy: {test_score:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model
    model_file = os.path.join(model_path, 'iris_model.pkl')
    joblib.dump(model, model_file)
    print(f"\nModel saved to: {model_file}")
    
    # Save metrics to JSON
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'n_estimators': n_estimators,
        'test_size': test_size,
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'test_accuracy': float(test_score)
    }
    
    metrics_file = os.path.join(model_path, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")
    
    print(f"\n{'='*50}")
    print("Training completed successfully!")
    print(f"{'='*50}")

if __name__ == '__main__':
    train_and_evaluate()