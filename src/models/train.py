"""
Model training pipeline with MLflow tracking.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any
import warnings

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from imblearn.over_sampling import SMOTE
from prefect import flow, task
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@task(name="load_data", retries=2)
def load_data(data_path: Path) -> tuple:
    """Load training and validation data."""
    logger.info(f"Loading data from {data_path}")
    
    train_df = pd.read_csv(data_path / "train.csv")
    val_df = pd.read_csv(data_path / "val.csv")
    
    # Separate features and target
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_val = val_df.drop('Class', axis=1)
    y_val = val_df['Class']
    
    logger.info(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")
    
    return X_train, y_train, X_val, y_val


@task(name="handle_imbalance")
def handle_imbalance(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """Handle class imbalance using SMOTE."""
    logger.info("Applying SMOTE for class balancing...")
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    logger.info(f"Original class distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Balanced class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
    
    return X_train_balanced, y_train_balanced


@task(name="train_model")
def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    params: Dict[str, Any]
) -> XGBClassifier:
    """Train XGBoost model."""
    logger.info("Training XGBoost model...")
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    logger.info("Model training complete")
    return model


@task(name="evaluate_model")
def evaluate_model(
    model: XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Dict[str, float]:
    """Evaluate model performance."""
    logger.info("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred_proba)
    }
    
    # Log confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Log classification report
    report = classification_report(y_val, y_pred)
    logger.info(f"Classification Report:\n{report}")
    
    # Log metrics
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return metrics


@flow(name="fraud_detection_training_pipeline")
def training_pipeline(
    data_path: str = "data/processed",
    model_params: Dict[str, Any] = None
):
    """Main training pipeline orchestrated by Prefect."""
    
    # Set default parameters
    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
    
    # Setup MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("fraud-detection")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(model_params)
        
        # Load data
        X_train, y_train, X_val, y_val = load_data(Path(data_path))
        
        # Handle imbalance
        X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train)
        
        # Train model
        model = train_model(X_train_balanced, y_train_balanced, model_params)
        
        # Evaluate model
        metrics = evaluate_model(model, X_val, y_val)
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="fraud-detection-xgboost"
        )
        
        # Save feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")
        
        logger.info(f"Model logged to MLflow with run_id: {mlflow.active_run().info.run_id}")
        
        return model, metrics


def main():
    """Main function to run training pipeline."""
    # Run the pipeline
    model, metrics = training_pipeline()
    
    logger.info("Training pipeline complete!")
    logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()