# Airflow DBSCAN Customer Clustering Pipeline

This project implements a complete machine learning pipeline using Apache Airflow for customer segmentation with DBSCAN clustering algorithm.

## 🏗️ Project Overview

The pipeline performs customer clustering analysis on credit card transaction data using DBSCAN (Density-Based Spatial Clustering of Applications with Noise), which automatically discovers clusters and identifies outliers without requiring pre-specified number of clusters.

## 📁 Project Structure

```
Lab_1/
├── README.md                   # Project documentation
├── docker-compose.yaml         # Airflow services configuration
├── dags/
│   └── airflow.py             # Main DAG definition
├── src/
│   ├── __init__.py
│   └── lab.py                 # ML pipeline functions
├── data/
│   ├── file.csv               # Training dataset (8,950 customers)
│   └── test.csv               # Test dataset (1 customer)
├── model/
│   └── dbscan_model.pkl       # Saved model output
├── logs/                      # Airflow execution logs
├── plugins/                   # Airflow plugins (empty)
└── config/                    # Airflow configuration
```

## 🎯 Key Features

### Algorithm: DBSCAN Clustering
- **Automatic cluster discovery** - No need to pre-specify number of clusters
- **Outlier detection** - Identifies anomalous customer behavior patterns
- **Density-based clustering** - Finds clusters of varying shapes and sizes
- **Noise handling** - Robust to outliers in financial data

### Data Processing
- **Feature Engineering** - Creates behavioral ratios and credit utilization metrics
- **RobustScaler** - Outlier-resistant data normalization
- **Missing Value Handling** - Robust preprocessing for real-world data

### Pipeline Architecture
- **Modular Design** - Clean separation between orchestration (DAG) and ML logic
- **4-Stage Pipeline** - Data loading → Feature engineering → Model training → Analysis
- **Error Handling** - Comprehensive logging and error recovery

## 📊 Dataset

**Training Data**: 8,950 customers with 18 features including:
- Financial metrics: BALANCE, PURCHASES, CREDIT_LIMIT, PAYMENTS
- Transaction patterns: PURCHASES_FREQUENCY, CASH_ADVANCE_TRX
- Behavioral indicators: TENURE, PRC_FULL_PAYMENT

**Engineered Features**:
- `BALANCE_TO_LIMIT_RATIO` - Credit utilization
- `AVAILABLE_CREDIT` - Remaining credit capacity
- `CASH_ADVANCE` - Cash advance usage (if available)

## 🚀 Setup Instructions

### Prerequisites
- Docker and Docker Compose
- 4GB+ RAM recommended
- 2+ CPU cores

### Installation

1. **Clone the repository**
   ```bash
   git clone <enter-this-repo-url>
   cd Lab_1
   ```

2. **Create environment file**
   ```bash
   echo "AIRFLOW_UID=$(id -u)" > .env
   echo "_AIRFLOW_WWW_USER_USERNAME=airflow" >> .env
   echo "_AIRFLOW_WWW_USER_PASSWORD=airflow" >> .env
   ```

3. **Initialize Airflow**
   ```bash
   docker-compose up airflow-init
   ```

4. **Start services**
   ```bash
   docker-compose up -d
   ```

5. **Access Airflow UI**
   - URL: http://localhost:8080
   - Username: `airflow`
   - Password: `airflow`

## 🔄 Pipeline Execution

### DAG: `Airflow-Lab_1`

The pipeline consists of 4 sequential tasks:

1. **explore_data_task**
   - Loads and analyzes customer dataset
   - Performs data quality assessment
   - Identifies missing values and data patterns

2. **feature_engineering_task**
   - Creates behavioral features and ratios
   - Applies RobustScaler normalization
   - Prepares data for DBSCAN clustering

3. **build_dbscan_model_task**
   - Tests multiple eps parameters (0.3, 0.5, 0.7, 1.0)
   - Selects optimal clustering configuration
   - Evaluates using silhouette score
   - Saves trained model

4. **analyze_clusters_and_predict_task**
   - Analyzes cluster characteristics
   - Assigns test customer to cluster or marks as outlier
   - Provides detailed clustering insights

### Running the Pipeline

1. Navigate to Airflow UI (http://localhost:8080)
2. Find the `Airflow-Lab_1` DAG
3. Toggle the DAG to "On" state
4. Click "Trigger DAG" to execute manually
5. Monitor task progress and view logs

## 📈 Expected Results

### DBSCAN Output
- **Clusters**: 2-6 customer segments (automatically determined)
- **Outliers**: 5-15% of customers flagged as anomalous behavior
- **Silhouette Score**: 0.2-0.6 (clustering quality metric)

### Business Insights
- **Cluster 0**: High-balance, low-activity customers
- **Cluster 1**: Active purchasers with moderate credit usage  
- **Cluster 2**: Cash advance users
- **Outliers**: Unusual spending patterns requiring investigation

## 🛠️ Technical Details

### Dependencies
- **Apache Airflow 2.5.1** - Workflow orchestration
- **scikit-learn** - DBSCAN implementation and preprocessing
- **pandas** - Data manipulation
- **numpy** - Numerical computations

### Docker Services
- **airflow-webserver** - Web UI (port 8080)
- **airflow-scheduler** - Task scheduling
- **airflow-worker** - Task execution
- **postgres** - Metadata database
- **redis** - Message broker

## 📝 Key Differences from K-Means

| Aspect | K-Means | DBSCAN |
|--------|---------|---------|
| Cluster Count | Must specify k | Automatic discovery |
| Outlier Handling | All points assigned | Explicit outlier detection |
| Cluster Shape | Spherical clusters | Arbitrary shapes |
| Parameter Sensitivity | k selection critical | eps parameter tuning |
| Business Value | Standard segmentation | Fraud/anomaly detection |

## 🔍 Monitoring and Debugging

### View Logs
```bash
# Task-specific logs
docker-compose logs airflow-scheduler
docker-compose logs airflow-worker

# Or check logs/ directory in project
```

### Common Issues
- **Memory**: Ensure 4GB+ RAM available
- **Permissions**: Run `echo "AIRFLOW_UID=$(id -u)" > .env`
- **Ports**: Ensure port 8080 is available

### Restart Services
```bash
docker-compose down
docker-compose up -d
```

## 🎓 Learning Outcomes

This lab demonstrates:
- **MLOps Pipeline Design** - End-to-end ML workflow automation
- **Advanced Clustering** - Beyond basic K-means to density-based methods
- **Production ML** - Containerized, scalable ML deployment
- **Data Engineering** - Feature engineering and preprocessing best practices
- **Outlier Detection** - Business-relevant anomaly identification

## 📧 Support

For issues or questions:
1. Check Airflow logs in the `logs/` directory
2. Verify Docker services are running: `docker-compose ps`
3. Ensure all dependencies are installed correctly

---
*This project was developed as part of IE-7374 MLOps coursework, demonstrating practical application of workflow orchestration and unsupervised machine learning techniques.*