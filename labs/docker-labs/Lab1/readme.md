# Docker ML Model Training Lab

A hands-on Docker lab demonstrating containerization of a machine learning workflow using scikit-learn and the Iris dataset. This lab teaches multi-stage builds, volume persistence, environment variable configuration, and health checks.

## Project Structure

```
docker-ml-lab/
├── .dockerignore       # Excludes unnecessary files from Docker build
├── Dockerfile          # Multi-stage build configuration
├── requirements.txt    # Python dependencies
└── src/
    └── main.py        # ML training script with evaluation
```

## File Descriptions

**Dockerfile**: Implements a multi-stage build with a builder stage for dependencies and a slim runtime stage for execution. Includes health checks and configurable environment variables.

**main.py**: Trains a Random Forest classifier on the Iris dataset with cross-validation, generates detailed evaluation metrics (classification report, confusion matrix), and persists both the model and metrics to a volume.

**requirements.txt**: Specifies Python package dependencies (scikit-learn, joblib, numpy).

**.dockerignore**: Prevents virtual environments, cache files, and other unnecessary files from being included in the Docker image.

## Prerequisites

- Docker installed and running
- Git Bash, PowerShell, or CMD terminal

## Quick Start

### 1. Build the Docker Image
```bash
docker build -t ml-trainer:v1 .
```

### 2. Create a Named Volume
```bash
docker volume create ml-models
```

### 3. Run the Training Container

**Git Bash (Windows):**
```bash
docker run --rm -v ml-models://models ml-trainer:v1
```

**PowerShell/CMD/Linux/Mac:**
```bash
docker run --rm -v ml-models:/models ml-trainer:v1
```

You should see training output including cross-validation scores, test accuracy, classification report, and confirmation that the model was saved.

### 4. Verify Model Persistence

**Git Bash:**
```bash
docker run --rm -v ml-models://models alpine ls -lh //models
```

**PowerShell/CMD/Linux/Mac:**
```bash
docker run --rm -v ml-models:/models alpine ls -lh /models
```

Expected output: `iris_model.pkl` and `metrics.json`

### 5. View Training Metrics

**Git Bash:**
```bash
docker run --rm -v ml-models://models alpine cat //models/metrics.json
```

**PowerShell/CMD/Linux/Mac:**
```bash
docker run --rm -v ml-models:/models alpine cat /models/metrics.json
```

## Advanced Usage

### Configure Hyperparameters

**Git Bash:**
```bash
docker run --rm -v ml-models://models \
  -e N_ESTIMATORS=200 \
  -e TEST_SIZE=0.3 \
  ml-trainer:v1
```

**PowerShell/CMD/Linux/Mac:**
```bash
docker run --rm -v ml-models:/models \
  -e N_ESTIMATORS=200 \
  -e TEST_SIZE=0.3 \
  ml-trainer:v1
```

### Interactive Debugging

```bash
docker run -it --rm -v ml-models:/models ml-trainer:v1 /bin/bash
```

Inside the container, explore files or test the model:
```bash
ls /models
python -c "import joblib; print(joblib.load('/models/iris_model.pkl'))"
```

## Git Bash Path Conversion Issue

If using Git Bash on Windows, paths like `/models` are converted to Windows paths. Use `//models` instead or set:
```bash
export MSYS_NO_PATHCONV=1
```

## Cleanup

```bash
# Remove volumes
docker volume rm ml-models

# Remove image
docker rmi ml-trainer:v1

# Clean up system
docker system prune
```
## Troubleshooting

**Build fails**: Ensure all files are in the correct directory structure.

**Path errors on Windows**: Use `//` prefix for paths in Git Bash or switch to PowerShell.

**Permission errors**: Run Docker commands with appropriate privileges or check Docker Desktop settings.
