# 🏠 House ## 📦 Project Structure

```
house-price-predictor/
├── configs/                # YAML-based configuration for models
├── data/                   # Raw and processed datasets
├── deployment/             # Deployment configurations
│   ├── kubernetes/         # Kubernetes manifests  
│   └── mlflow/             # MLflow tracking setup
├── models/trained/         # Trained ML models and preprocessors
├── notebooks/              # Jupyter notebooks for experimentation
├── src/
│   ├── api/                # FastAPI backend service
│   ├── data/               # Data cleaning and preprocessing scripts
│   ├── features/           # Feature engineering pipeline
│   └── models/             # Model training and evaluation
├── streamlit_app/          # Streamlit frontend service
├── docker-compose.yml      # Complete application orchestration
├── Dockerfile.api          # FastAPI backend container
├── ARCHITECTURE_OVERVIEW.md # Detailed system documentation
├── pyproject.toml          # Modern Python project configuration
└── README.md               # You're here!
``` An MLOps Learning Project

Welcome to the **House Price Predictor** project! This is a real-world, end-to-end MLOps use case designed to help you master the art of building and operationalizing machine learning pipelines.

You'll start from raw data and move through data preprocessing, feature engineering, experimentation, model tracking with MLflow, and optionally using Jupyter for exploration – all while applying industry-grade tooling.

> 🚀 **Want to master MLOps from scratch?**  
Check out the [MLOps Bootcamp at School of DevOps](https://schoolofdevops.com) to level up your skills.

---

## 📦 Project Structure

```
house-price-predictor/
├── configs/                # YAML-based configuration for models
├── data/                   # Raw and processed datasets
├── deployment/
│   └── mlflow/             # Docker Compose setup for MLflow
├── models/                 # Trained models and preprocessors
├── notebooks/              # Optional Jupyter notebooks for experimentation
├── src/
│   ├── data/               # Data cleaning and preprocessing scripts
│   ├── features/           # Feature engineering pipeline
│   ├── models/             # Model training and evaluation
├── requirements.txt        # Python dependencies
└── README.md               # You’re here!
```

---

## 🛠️ Setting up Learning/Development Environment

To begin, ensure the following tools are installed on your system:

- [Python 3.11](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)
- [Visual Studio Code](https://code.visualstudio.com/) or your preferred editor
- [UV – Python package and environment manager](https://github.com/astral-sh/uv)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) **or** [Podman Desktop](https://podman-desktop.io/)

---

## 🚀 Preparing Your Environment

1. **Fork this repo** on GitHub.

2. **Clone your forked copy:**

   ```bash
   # Replace xxxxxx with your GitHub username or org
   git clone https://github.com/xxxxxx/house-price-predictor.git
   cd house-price-predictor
   ```

3. **Setup Python Virtual Environment using UV:**

   ```bash
   uv venv --python python3.11
   source .venv/bin/activate
   ```

4. **Install dependencies:**

   ```bash
   # Install all dependencies (ML pipeline + API + Frontend + Dev tools)
   uv pip install -e ".[all]"
   
   # Or install only what you need:
   # uv pip install -e "."          # Core ML pipeline only
   # uv pip install -e ".[api]"     # Core + API dependencies
   # uv pip install -e ".[dev]"     # Core + Development tools
   ```

---

## 📊 Setup MLflow for Experiment Tracking

To track experiments and model runs:

```bash
cd deployment/mlflow
docker compose -f mlflow-docker-compose.yml up -d
docker compose ps
```

> 🐧 **Using Podman?** Use this instead:

```bash
podman compose -f mlflow-docker-compose.yml up -d
podman compose ps
```

Access the MLflow UI at [http://localhost:5555](http://localhost:5555)

---

## 📒 Using JupyterLab (Optional)

If you prefer an interactive experience, launch JupyterLab with:

```bash
uv python -m jupyterlab
# or
python -m jupyterlab
```

---

## ⚡ Quick Start for Existing Users

**Already have trained models?** Skip the ML development and go straight to deployment:

```bash
# 1. Verify you have the required model files
ls models/trained/
# Should show: house_price_model.pkl  preprocessor.pkl

# 2. Start the application
docker-compose up
```

**Services will be available at:**
- 🌐 **Streamlit App**: http://localhost:8501
- 🔌 **FastAPI Backend**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs

**Test the API:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sqft": 2000,
    "bedrooms": 3,
    "bathrooms": 2,
    "location": "suburban",
    "year_built": 1990,
    "condition": "Good"
  }'
```

---

## 🎯 Complete Workflow Overview

This project follows a **two-phase approach**:

1. **🔬 ML Development Phase**: Build and train your model locally  
2. **🚀 Deployment Phase**: Deploy the trained model using Docker

Choose your path based on your situation:

### 🆕 **New Users** (Starting from Scratch)
If you're cloning this repository for the first time:
- Follow **Phase 1** (ML Development) first to generate trained models
- Then proceed to **Phase 2** (Docker Deployment)

### ✅ **Existing Users** (Already Have Trained Models) 
If you already have `models/trained/house_price_model.pkl` and `preprocessor.pkl`:
- Skip directly to **Phase 2** (Docker Deployment)
- Just run `docker-compose up`

---

## 🔁 Phase 1: ML Development Workflow
> **🆕 For New Users Only** - Skip if you already have trained models

### 🧹 Step 1: Data Processing

Clean and preprocess the raw housing dataset:

```bash
python src/data/run_processing.py   --input data/raw/house_data.csv   --output data/processed/cleaned_house_data.csv
```

---

### 🧠 Step 2: Feature Engineering

Apply transformations and generate features:

```bash
python src/features/engineer.py   --input data/processed/cleaned_house_data.csv   --output data/processed/featured_house_data.csv   --preprocessor models/trained/preprocessor.pkl
```

---

### 📈 Step 3: Modeling & Experimentation

Train your model and log everything to MLflow:

```bash
python src/models/train_model.py   --config configs/model_config.yaml   --data data/processed/featured_house_data.csv   --models-dir models/trained   --mlflow-tracking-uri http://localhost:5555
```

> ✅ **Important**: This step generates the trained model files (`house_price_model.pkl` and `preprocessor.pkl`) in `models/trained/` that are required for the Docker deployment below.

---


## 🚀 Phase 2: Containerized Deployment with Docker

After completing the ML workflow above, you can deploy the complete application using Docker Compose.

### Prerequisites for Docker Deployment
- Docker Desktop or Podman Desktop installed
- Trained models available in `models/trained/` (from Step 3 above)

### 🚀 Quick Start with Docker

**Launch the complete MLOps application:**

```bash
docker-compose up
```

This will start both services:
- **FastAPI Backend**: http://localhost:8000
- **Streamlit Frontend**: http://localhost:8501

### 🌐 Accessing the Application

1. **Web Interface**: Open http://localhost:8501 in your browser
   - Interactive web form for house price predictions
   - Real-time results with confidence intervals

2. **API Documentation**: Visit http://localhost:8000/docs
   - Interactive Swagger UI for API testing
   - Complete endpoint documentation

3. **Health Check**: http://localhost:8000/health
   - Verify that models are loaded successfully

### 🧪 Testing the API Directly

Test predictions using curl:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sqft": 2000,
    "bedrooms": 3,
    "bathrooms": 2,
    "location": "suburban",
    "year_built": 1990,
    "condition": "Good"
  }'
```

**Expected Response:**
```json
{
  "predicted_price": 460579.86,
  "confidence_interval": [414521.87, 506637.85],
  "features_importance": {},
  "prediction_time": "2025-11-02T00:39:25.050877"
}
```

### 🛠️ Docker Management Commands

```bash
# Stop services
docker-compose down

# View logs
docker-compose logs

# Rebuild images (if code changes)
docker-compose build

# Run in background
docker-compose up -d
```

### 🔍 Troubleshooting Docker Deployment

**Problem**: Docker containers fail to start
- **Solution**: Ensure trained models exist in `models/trained/` by completing Phase 1 first

**Problem**: API returns errors
- **Solution**: Check that both `house_price_model.pkl` and `preprocessor.pkl` are present

**Problem**: Port conflicts
- **Solution**: Make sure ports 8000 and 8501 are not already in use

**Problem**: "Model not found" errors
- **Solution**: Verify the model training step completed successfully and generated the `.pkl` files

--- 

## 📋 Quick Reference

### Complete Setup from Scratch
```bash
# 1. ML Development Phase
python src/data/run_processing.py --input data/raw/house_data.csv --output data/processed/cleaned_house_data.csv
python src/features/engineer.py --input data/processed/cleaned_house_data.csv --output data/processed/featured_house_data.csv --preprocessor models/trained/preprocessor.pkl
python src/models/train_model.py --config configs/model_config.yaml --data data/processed/featured_house_data.csv --models-dir models/trained --mlflow-tracking-uri http://localhost:5555

# 2. Docker Deployment Phase
docker-compose up
```

### Verify Everything Works
- MLflow UI: http://localhost:5555 (if running)
- Streamlit App: http://localhost:8501
- FastAPI Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

--- 


## 🧠 Learn More About MLOps

This project is part of the [**MLOps Bootcamp**](https://schoolofdevops.com) at School of DevOps, where you'll learn how to:

- Build and track ML pipelines
- Containerize and deploy models
- Automate training workflows using GitHub Actions or Argo Workflows
- Apply DevOps principles to Machine Learning systems

🔗 [Get Started with MLOps →](https://schoolofdevops.com)

---

## 🤝 Contributing

We welcome contributions, issues, and suggestions to make this project even better. Feel free to fork, explore, and raise PRs!

---

Happy Learning!  
— Team **School of DevOps**