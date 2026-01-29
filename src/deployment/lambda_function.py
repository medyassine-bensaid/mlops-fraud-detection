"""
AWS Lambda function handler for fraud detection predictions.
"""
import json
import logging
import os
from typing import Dict, Any

import boto3
import mlflow
import pandas as pd
import numpy as np

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global variables for model caching
MODEL = None
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "mlops-fraud-detection-data-models")


def load_model():
    """Load model from MLflow or S3."""
    global MODEL
    
    if MODEL is None:
        logger.info("Loading model...")
        
        try:
            # Try loading from MLflow
            model_uri = "models:/fraud-detection-xgboost/production"
            MODEL = mlflow.xgboost.load_model(model_uri)
            logger.info("Model loaded from MLflow")
        except Exception as e:
            logger.warning(f"Failed to load from MLflow: {e}")
            
            # Fallback: load from S3
            try:
                s3 = boto3.client('s3')
                model_path = "/tmp/model.pkl"
                s3.download_file(MODEL_BUCKET, "models/model.pkl", model_path)
                import joblib
                MODEL = joblib.load(model_path)
                logger.info("Model loaded from S3")
            except Exception as s3_error:
                logger.error(f"Failed to load model from S3: {s3_error}")
                raise
    
    return MODEL


def preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess input data for prediction."""
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Ensure all required features are present
    # Add feature engineering if needed
    
    return df


def handler(event, context):
    """
    Lambda handler function for predictions.
    
    Expected input format:
    {
        "Time": 0.0,
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        ...
        "V28": -0.0210530534538215,
        "Amount": 149.62
    }
    """
    try:
        # Parse input
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event
        
        logger.info(f"Received prediction request: {body}")
        
        # Load model
        model = load_model()
        
        # Preprocess input
        input_df = preprocess_input(body)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'fraud_probability': float(prediction_proba[1]),
            'confidence': float(max(prediction_proba)),
            'status': 'success'
        }
        
        logger.info(f"Prediction result: {response}")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response)
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'status': 'error',
                'message': str(e)
            })
        }


# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {
        "Time": 0.0,
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        "V3": 2.53634673796914,
        "V4": 1.37815522427443,
        "V5": -0.338320769942518,
        "V6": 0.462387777762292,
        "V7": 0.239598554061257,
        "V8": 0.0986979012610507,
        "V9": 0.363786969611213,
        "V10": 0.0907941719789316,
        "V11": -0.551599533260813,
        "V12": -0.617800855762348,
        "V13": -0.991389847235408,
        "V14": -0.311169353699879,
        "V15": 1.46817697209427,
        "V16": -0.470400525259478,
        "V17": 0.207971241929242,
        "V18": 0.0257905801985591,
        "V19": 0.403992960255733,
        "V20": 0.251412098239705,
        "V21": -0.018306777944153,
        "V22": 0.277837575558899,
        "V23": -0.110473910188767,
        "V24": 0.0669280749146731,
        "V25": 0.128539358273528,
        "V26": -0.189114843888824,
        "V27": 0.133558376740387,
        "V28": -0.0210530534538215,
        "Amount": 149.62
    }
    
    result = handler(test_event, None)
    print(json.dumps(result, indent=2))