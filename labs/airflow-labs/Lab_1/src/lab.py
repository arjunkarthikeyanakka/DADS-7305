import pandas as pd
from sklearn.preprocessing import RobustScaler  
from sklearn.cluster import DBSCAN  
from sklearn.metrics import silhouette_score
import pickle
import os
import base64
import numpy as np

def load_and_explore_data():
    """
    Loads data from a CSV file, performs basic exploration, and returns serialized data.
    """
    print("Loading and exploring data...")
    data_path = os.path.join("/opt/airflow", "data", "file.csv")
    df = pd.read_csv(data_path)
    
    print(f"Data shape: {df.shape}")
    print(f"Data columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")

def feature_engineering_preprocessing(data_b64: str):
    """
    Performs feature engineering and preprocessing for DBSCAN clustering.
    """
    print("Starting feature engineering and preprocessing...")
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)
    
    # Clean data
    df = df.dropna()
    
    # Feature engineering - create new features
    if 'BALANCE' in df.columns and 'CREDIT_LIMIT' in df.columns:
        df['BALANCE_TO_LIMIT_RATIO'] = df['BALANCE'] / (df['CREDIT_LIMIT'] + 1)
        df['AVAILABLE_CREDIT'] = df['CREDIT_LIMIT'] - df['BALANCE']
        print("Created engineered features: BALANCE_TO_LIMIT_RATIO, AVAILABLE_CREDIT")
    
    # Select features for clustering
    feature_cols = ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'BALANCE_TO_LIMIT_RATIO', 'AVAILABLE_CREDIT']
    
    # Add CASH_ADVANCE if available
    if 'CASH_ADVANCE' in df.columns:
        feature_cols.append('CASH_ADVANCE')
        print("Added CASH_ADVANCE feature")
    
    clustering_data = df[feature_cols]
    print(f"Features shape: {clustering_data.shape}")
    print(f"Selected features: {feature_cols}")
    
    # Use RobustScaler instead of StandardScaler (better for outliers)
    scaler = RobustScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)
    
    print("Applied RobustScaler preprocessing")
    
    # Package data for next task
    processed_data = {
        'scaled_data': clustering_data_scaled,
        'feature_names': feature_cols,
        'scaler': scaler,
        'original_data': df
    }
    
    serialized_data = pickle.dumps(processed_data)
    print("Feature engineering completed successfully")
    return base64.b64encode(serialized_data).decode("ascii")

def build_dbscan_model(data_b64: str, model_filename: str):
    """
    Builds a DBSCAN clustering model (replaces K-means).
    """
    print("Building DBSCAN clustering model...")
    data_bytes = base64.b64decode(data_b64)
    processed_data = pickle.loads(data_bytes)
    
    scaled_data = processed_data['scaled_data']
    feature_names = processed_data['feature_names']
    
    print(f"Clustering {len(scaled_data)} samples with {len(feature_names)} features")
    
    # Test different eps values to find good clustering
    eps_values = [0.3, 0.5, 0.7, 1.0]
    best_model = None
    best_score = -1
    best_eps = 0.5
    
    print("Testing different eps values for DBSCAN...")
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(scaled_data)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"eps={eps}: {n_clusters} clusters, {n_noise} noise points")
        
        # Calculate silhouette score if we have valid clusters
        if n_clusters > 1:
            sil_score = silhouette_score(scaled_data, labels)
            print(f"  Silhouette score: {sil_score:.3f}")
            
            if sil_score > best_score:
                best_score = sil_score
                best_model = dbscan
                best_eps = eps
        else:
            print("  Not enough clusters for evaluation")
    
    if best_model is None:
        # Fallback to a reasonable eps
        print("Using fallback eps=0.5")
        best_model = DBSCAN(eps=0.5, min_samples=5)
        labels = best_model.fit_predict(scaled_data)
        best_eps = 0.5
    else:
        labels = best_model.fit_predict(scaled_data)
    
    # Analyze final results
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Final DBSCAN model:")
    print(f"  eps: {best_eps}")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    print(f"  Silhouette score: {best_score:.3f}")
    
    # Save the model
    output_dir = os.path.join("/opt/airflow", "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, model_filename)
    
    model_data = {
        'model': best_model,
        'labels': labels,
        'eps': best_eps,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette_score': best_score if best_score > -1 else 0,
        'feature_names': feature_names,
        'scaler': processed_data['scaler']
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"DBSCAN model saved to: {output_path}")
    
    return {
        'n_clusters': int(n_clusters),
        'n_noise': int(n_noise),
        'eps_used': float(best_eps),
        'silhouette_score': float(best_score) if best_score > -1 else 0.0
    }

def analyze_clusters_and_predict(model_filename: str, model_results: dict):
    """
    Analyzes DBSCAN clusters and makes predictions on test data.
    """
    print("Analyzing DBSCAN clusters and making predictions...")
    
    # Load the saved model
    model_path = os.path.join("/opt/airflow", "model", model_filename)
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    labels = model_data['labels']
    feature_names = model_data['feature_names']
    scaler = model_data['scaler']
    n_clusters = model_data['n_clusters']
    n_noise = model_data['n_noise']
    
    print(f"DBSCAN Results Summary:")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise}")
    print(f"  Parameters: eps={model_data['eps']}, min_samples=5")
    
    # Analyze cluster sizes
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            count = list(labels).count(label)
            print(f"  Noise/Outliers: {count} customers")
        else:
            count = list(labels).count(label)
            print(f"  Cluster {label}: {count} customers")
    
    # Load test data for prediction
    try:
        test_data_path = os.path.join("/opt/airflow", "data", "test.csv")
        df_test = pd.read_csv(test_data_path)
        
        print(f"Processing test customer...")
        
        # Apply same feature engineering
        df_test_processed = df_test.copy()
        df_test_processed['BALANCE_TO_LIMIT_RATIO'] = df_test_processed['BALANCE'] / (df_test_processed['CREDIT_LIMIT'] + 1)
        df_test_processed['AVAILABLE_CREDIT'] = df_test_processed['CREDIT_LIMIT'] - df_test_processed['BALANCE']
        
        # Handle missing features
        for feature in feature_names:
            if feature not in df_test_processed.columns:
                df_test_processed[feature] = 0
                print(f"Added missing feature {feature} with default value 0")
        
        test_features = df_test_processed[feature_names]
        test_scaled = scaler.transform(test_features)
        
        # DBSCAN prediction (assign to nearest cluster or mark as noise)
        # For DBSCAN, we need to use the training data to make predictions
        from sklearn.neighbors import NearestNeighbors
        
        # Find nearest neighbors from training data
        nbrs = NearestNeighbors(n_neighbors=1).fit(model.components_)  # Use core points
        distances, indices = nbrs.kneighbors(test_scaled)
        
        # Get the cluster of the nearest core point
        nearest_core_idx = indices[0][0]
        prediction = labels[model.core_sample_indices_[nearest_core_idx]]
        distance_to_cluster = distances[0][0]
        
        print(f"Test customer prediction:")
        if prediction == -1:
            print(f"  Assigned to: Noise/Outlier")
            print(f"  Distance to nearest cluster: {distance_to_cluster:.3f}")
        else:
            print(f"  Assigned to: Cluster {prediction}")
            print(f"  Distance to cluster center: {distance_to_cluster:.3f}")
        
        return {
            'cluster_prediction': int(prediction),
            'is_outlier': bool(prediction == -1),
            'distance_to_cluster': float(distance_to_cluster),
            'n_clusters_found': int(n_clusters),
            'n_noise_points': int(n_noise)
        }
        
    except Exception as e:
        error_msg = f"Error making prediction: {str(e)}"
        print(f"Error: {error_msg}")
        return {
            'error': error_msg,
            'n_clusters_found': int(n_clusters),
            'n_noise_points': int(n_noise)
        }