import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import logging
import os

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.makedirs('plots', exist_ok=True)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("house_price_prediction")

def generate_house_data(n_samples=1000):
    '''
        Generates random housing data 
    '''
    np.random.seed(42)
    
    size = np.random.uniform(800, 4000, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    age = np.random.uniform(0, 50, n_samples)
    location_score = np.random.uniform(1, 10, n_samples)
    
    price = (
        200 * size + 
        50000 * bedrooms + 
        -1000 * age + 
        30000 * location_score + 
        np.random.normal(0, 50000, n_samples)
    )
    
    return pd.DataFrame({
        'size_sqft': size,
        'bedrooms': bedrooms,
        'age_years': age,
        'location_score': location_score,
        'price': price
    })

def train_model(model, model_name, X_train, X_test, y_train, y_test, params=None):
    '''
        Takes a model, trains and evaluates it.
        Logs everything using mlflow.
        Returns the trained model, its R2 score and the run ID.
    '''
    with mlflow.start_run(run_name=model_name) as run:
        if params:
            mlflow.log_params(params)
        
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_mae", test_mae)
        
        signature = mlflow.models.infer_signature(X_train, y_pred_train)
        input_example = X_train[:5]
        
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'{model_name} - Predictions vs Actual')
        plt.tight_layout()
        plt.savefig(f'plots/{model_name}_predictions.png')
        mlflow.log_artifact(f'plots/{model_name}_predictions.png')
        plt.close()
        
        if hasattr(model, 'feature_importances_'):
            feature_names = ['size_sqft', 'bedrooms', 'age_years', 'location_score']
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            mlflow.log_params({f"feature_importance_{k}": v for k, v in importance_dict.items()})
        
        logger.info(f"{model_name} - Train RMSE: ${train_rmse:,.2f}, Test RMSE: ${test_rmse:,.2f}, Test R²: {test_r2:.4f}")
        
        return model, test_r2, run.info.run_id

def main():
    logger.info("Starting house price prediction experiment")
    
    df = generate_house_data(n_samples=5000)
    logger.info(f"Generated {len(df)} samples")
    logger.info(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    
    X = df.drop('price', axis=1)
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Training models...")
    
    results = []
    
    logger.info("[1/5] Training Linear Regression")
    _, r2, run_id = train_model(
        LinearRegression(),
        "Linear_Regression",
        X_train_scaled, X_test_scaled, y_train, y_test,
        params={"model_type": "linear"}
    )
    results.append(("Linear_Regression", r2, run_id))
    
    logger.info("[2/5] Training Ridge Regression")
    _, r2, run_id = train_model(
        Ridge(alpha=1.0),
        "Ridge_Regression",
        X_train_scaled, X_test_scaled, y_train, y_test,
        params={"model_type": "ridge", "alpha": 1.0}
    )
    results.append(("Ridge_Regression", r2, run_id))
    
    logger.info("[3/5] Training Lasso Regression")
    _, r2, run_id = train_model(
        Lasso(alpha=1.0),
        "Lasso_Regression",
        X_train_scaled, X_test_scaled, y_train, y_test,
        params={"model_type": "lasso", "alpha": 1.0}
    )
    results.append(("Lasso_Regression", r2, run_id))
    
    logger.info("[4/5] Training Random Forest")
    _, r2, run_id = train_model(
        RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
        "Random_Forest_Light",
        X_train, X_test, y_train, y_test,
        params={"model_type": "random_forest", "n_estimators": 50, "max_depth": 10}
    )
    results.append(("Random_Forest_Light", r2, run_id))
    
    logger.info("[5/5] Training Gradient Boosting")
    _, r2, run_id = train_model(
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "Gradient_Boosting",
        X_train, X_test, y_train, y_test,
        params={"model_type": "gradient_boosting", "n_estimators": 100, "learning_rate": 0.1}
    )
    results.append(("Gradient_Boosting", r2, run_id))
    
    best_model_name, best_r2, best_run_id = max(results, key=lambda x: x[1])
    logger.info(f"Best model: {best_model_name} with R² = {best_r2:.4f}")
    
    model_uri = f"runs:/{best_run_id}/model"
    registered_model = mlflow.register_model(model_uri, "house_price_predictor")
    
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        "house_price_predictor",
        "champion",
        registered_model.version
    )
    
    logger.info(f"Registered model 'house_price_predictor' version {registered_model.version} as champion")
    logger.info("Experiments completed. Run 'mlflow ui' to view results")

if __name__ == "__main__":
    main()