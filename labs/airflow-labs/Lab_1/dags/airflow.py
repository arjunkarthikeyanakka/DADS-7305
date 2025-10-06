import sys
import os

sys.path.append('/opt/airflow/src')

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Import our custom ML functions from the lab module
from lab import (
    load_and_explore_data,
    feature_engineering_preprocessing,
    build_dbscan_model,   
    analyze_clusters_and_predict   
)

# Define default arguments for the DAG
default_args = {
    'owner': 'dbscan_clustering_student',
    'start_date': datetime(2025, 1, 15),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False,
}

# Create the DAG instance
with DAG(
    dag_id='Airflow-Lab_1',  
    default_args=default_args,
    description='Customer Clustering using DBSCAN with Outlier Detection',  
    schedule_interval=None,  
    catchup=False,
    tags=['dbscan', 'clustering', 'outlier-detection', 'customer-segmentation'],  
    max_active_runs=1,  
) as dag:

    # Task 1: Load and explore the dataset
    explore_data_task = PythonOperator(
        task_id='explore_data_task',
        python_callable=load_and_explore_data,
        doc_md="""
        ## Data Loading and Exploration
        
        This task:
        - Loads customer data from `/opt/airflow/data/file.csv`
        - Performs basic data exploration and quality checks
        - Returns serialized data for preprocessing
        """,
    )

    # Task 2: Feature engineering and preprocessing
    feature_engineering_task = PythonOperator(
        task_id='feature_engineering_task',
        python_callable=feature_engineering_preprocessing,
        op_args=[explore_data_task.output],
        doc_md="""
        ## Feature Engineering and Preprocessing
        
        This task:
        - Creates behavioral features (ratios, balances)
        - Applies RobustScaler for outlier-resistant normalization
        - Prepares features optimized for DBSCAN clustering
        """,
    )

    # Task 3: Build DBSCAN clustering model
    build_dbscan_task = PythonOperator(
        task_id='build_dbscan_model_task',  
        python_callable=build_dbscan_model,
        op_args=[feature_engineering_task.output, "dbscan_model.pkl"],  
        doc_md="""
        ## DBSCAN Clustering Model
        
        This task:
        - Tests different eps parameters for optimal clustering
        - Builds density-based clusters (no need to specify k)
        - Automatically detects outliers and noise points
        - Evaluates clustering quality using silhouette score
        """,
    )

    # Task 4: Analyze clusters and make predictions
    analyze_clusters_task = PythonOperator(
        task_id='analyze_clusters_and_predict_task',  
        python_callable=analyze_clusters_and_predict,
        op_args=["dbscan_model.pkl", build_dbscan_task.output],
        doc_md="""
        ## Cluster Analysis and Customer Prediction
        
        This task:
        - Analyzes DBSCAN cluster characteristics
        - Identifies customer segments and outliers
        - Assigns test customers to clusters or marks as outliers
        - Provides distance-based confidence measures
        """,
    )

    # Define task dependencies
    explore_data_task >> feature_engineering_task >> build_dbscan_task >> analyze_clusters_task