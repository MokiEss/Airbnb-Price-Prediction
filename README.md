# 🏡 Airbnb Price Prediction

This project predicts Airbnb listing prices using machine learning, specifically the XGBoost algorithm. It includes a complete workflow from data preprocessing to model training, evaluation, and deployment in a Docker container.

## 🚀 Features

- Predict Airbnb listing prices using XGBoost.
- Clean and preprocess Airbnb dataset.
- Evaluate model performance using standard regression metrics.
- Containerized with Docker for easy deployment.


## 🔧 Setup Instructions

For running the project:

- Clone the repository.
- Make sure you have docker and docker_compose installed in your system.
- Change directory to Airbnb-Price-Prediction:
 ```bash
git clone https://github.com/MokiESS/Airbnb-Price-Prediction.git
cd Airbnb-Price-Prediction
```

- For starting the project run:
```bash
docker-compose up --build
```
## 📊 Model Performance

- Algorithm: XGBoost Regressor
- Evaluation Metrics: MAE, RMSE, R²
- Cross-Validation: Yes (e.g., K-Fold)

## 📦 Requirements

- Python 3.8+
- xgboost
- pandas
- scikit-learn
- numpy
