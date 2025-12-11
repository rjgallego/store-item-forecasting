# 🚀 Store Item Demand Forecasting Project

A full end-to-end machine learning pipeline for demand forecasting, including data validation, feature engineering, model training, rolling backtesting, and Dockerized deployment.

## 📌 1. Project Summary

This project implements a complete forecasting system for the Store Item Demand Forecasting dataset (Kaggle). It follows real-world machine learning engineering practices:
- **Raw data ingestion & validation** (pandera schemas)
- **Feature engineering pipeline** (calendar features, lag features, rolling statistics, cyclic seasonality)
- **Model training** using **LightGBM** and **CatBoost**
- **Rolling-origin backtesting** to evaluate performance over time
- **Baselines** including naive-1, naive-7, and naive-28
- **Batch inference CLI** for generating predictions
- **Dockerized deployment** for reproducible forecasting anywhere

The final system demonstrates strong forecasting performance (~4.08 MAE with CatBoost) and is structured the same way production ML systems are built in industry.

--- 

## 📂 2. Project Structure
forecast-studio/

│

├── data/

│   ├── raw/                 # Original Kaggle dataset (not versioned)

│   ├── processed/           # Engineered features (parquet files)

│

├── models/

│   └── catboost_baseline.pkl   # Saved model artifacts (ignored by Git)

│

├── reports/

│   ├── backtest_rolling_windows.csv

│   ├── backtest_catboost_rolling_windows.csv

│   └── predictions.csv        # Saved forecast outputs

│

├── src/

│   ├── features/

│   │   ├── config.py              # Global configuration paths & constants

│   │   ├── schema.py          # Pandera schemas for raw & processed data

│   │   ├── make_features.py   # Main feature engineering pipeline

│   │

│   ├── models/

│   │   ├── baseline.py        # LightGBM training logic

│   │   ├── catboost_baseline.py

│   │   └── predict_catboost.py # CLI prediction tool

│   │

│   ├── eval/

│   │   ├── backtest.py        # Rolling-origin backtesting (LightGBM)

│   │   └── backtest_catboost.py

│

├── Dockerfile                 # Dockerized forecasting environment

├── .dockerignore

├── .gitignore

├── requirements.txt

└── README.md

## ▶️ 3. How to Download & Run the Project
### Step 1 — Clone the repo

*git clone https://github.com/your-username/repo-name.git*

*cd repo-name*

### Step 2 — Create a virtual environment

*python -m venv venv*

*source venv/bin/activate*      # Mac/Linux

*venv\Scripts\activate*         # Windows

### Step 3 — Install dependencies

*pip install -r requirements.txt*

### Step 4 — Add the dataset

**Download the Store Item Demand Forecasting dataset from Kaggle and place:**

train.csv → data/raw/train.csv

test.csv  → data/raw/test.csv

### Step 5 — Generate features

*python -m src.features.make_features*


This will output:

data/processed/train_features.parquet

data/processed/test_features.parquet

### Step 6 — Train the model

**CatBoost (best-performing):**

*python -m src.models.train_catboost*


**LightGBM (baseline):**

*python -m src.models.train_baseline*

### Step 7 — Run rolling backtests

**CatBoost:**

*python -m src.eval.backtest_catboost*


**LightGBM:**

*python -m src.eval.backtest*


Backtest results appear in reports/.

### Step 8 — Generate predictions (batch inference)

*python -m src.models.predict_catboost* \

  *--model-path models/catboost_baseline.pkl* \
  
  *--input-path data/processed/test_features.parquet* \
  
  *--output-path reports/predictions.csv*

### 🐳 4. Running the Project with Docker

Build the Docker image

*docker build -t store-forecast .*


Run predictions inside Docker

docker run --rm -v "${PWD}:/app" store-forecast \

  --model-path models/catboost_baseline.pkl \
  
  --input-path data/processed/test_features.parquet \
  
  --output-path reports/docker_predictions.csv


Because the Dockerfile includes:

ENTRYPOINT ["python", "-m", "src.models.predict_catboost"]


the CLI works like a native tool inside Docker.

### 📊 5. Model Performance

Average performance across multiple backtest windows:


Model,	MAE (avg),	RMSE (avg)

CatBoost,	~4.08,	~5.30

LightGBM,	~4.11,	~5.33

Naive-7,	~6.7–9.0	higher

Naive-28,	~7.0–10	higher


The model consistently outperforms:

- naive-1 baseline
- weekly seasonal naive (lag 7)
- monthly seasonal naive (lag 28)

This confirms strong and stable forecasting behavior.

## 🧩 6. Key Features of This Project

✔ Pandera schemas for robust data validation

✔ Daily continuity checks to avoid broken time series

✔ Lag features + rolling mean & std windows

✔ Cyclic seasonal features (sin/cos encodings)

✔ Holiday proximity features

✔ Global forecasting model (shared across all store-item series)

✔ Rolling-origin time-series backtesting

✔ Model comparison (LightGBM vs CatBoost)

✔ Dockerized CLI for batch predictions

✔ Strong modular design for extensibility

## 🧭 7. Future Improvements (Roadmap)

- Add hyperparameter tuning (Optuna or Ray Tune)
- Deploy a FastAPI prediction server
- Push Docker image to GitHub Container Registry
- Build a Streamlit dashboard for visualizing forecasts

## 👨‍💻 8. Author

Created by: Rheanna Pena

[GitHub](https://github.com/rjgallego)

[LinkedIn](https://www.linkedin.com/in/rheanna-pena-aa0007110/)
