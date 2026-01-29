"""
Integration tests for the entire ML pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.preprocess import DataPreprocessor
from models.train import train_model, evaluate_model


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataset(temp_data_dir):
    """Create a complete sample dataset."""
    np.random.seed(42)
    n_samples = 5000
    
    # Create synthetic data
    data = {
        'Time': np.random.rand(n_samples) * 172800,
        'Amount': np.random.rand(n_samples) * 1000,
        'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    }
    
    # Add V1-V28 features
    for i in range(1, 29):
        # Make features slightly correlated with Class for realistic testing
        if i <= 10:
            data[f'V{i}'] = np.random.randn(n_samples) + data['Class'] * 2
        else:
            data[f'V{i}'] = np.random.randn(n_samples)
    
    df = pd.DataFrame(data)
    
    # Save raw data
    raw_dir = temp_data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_dir / "creditcard.csv", index=False)
    
    return temp_data_dir


class TestEndToEndPipeline:
    """Test the complete ML pipeline."""
    
    def test_complete_pipeline(self, sample_dataset, temp_data_dir):
        """Test the complete pipeline from data to model."""
        
        # Step 1: Data Preprocessing
        preprocessor = DataPreprocessor()
        
        # Load raw data
        raw_data = preprocessor.load_data(sample_dataset / "raw" / "creditcard.csv")
        assert len(raw_data) == 5000
        
        # Preprocess
        processed_data = preprocessor.preprocess(raw_data)
        assert 'Amount_scaled' in processed_data.columns
        assert 'Amount' not in processed_data.columns
        
        # Split data
        train, val, test = preprocessor.split_data(processed_data)
        
        # Save processed data
        processed_dir = temp_data_dir / "processed"
        preprocessor.save_data(train, val, test, processed_dir)
        
        # Verify saved files
        assert (processed_dir / "train.csv").exists()
        assert (processed_dir / "val.csv").exists()
        assert (processed_dir / "test.csv").exists()
        
        # Step 2: Model Training
        X_train = train.drop('Class', axis=1)
        y_train = train['Class']
        
        model_params = {
            'n_estimators': 50,  # Reduced for testing
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        model = train_model(X_train, y_train, model_params)
        assert model is not None
        
        # Step 3: Model Evaluation
        X_val = val.drop('Class', axis=1)
        y_val = val['Class']
        
        metrics = evaluate_model(model, X_val, y_val)
        
        # Check that metrics are reasonable
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        # All metrics should be between 0 and 1
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} is out of range: {value}"
        
        # Accuracy should be reasonably high (>70% even with random features)
        assert metrics['accuracy'] > 0.7
        
        # Step 4: Predictions on test set
        X_test = test.drop('Class', axis=1)
        y_test = test['Class']
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})
        
        # Test prediction probabilities
        pred_proba = model.predict_proba(X_test)
        assert pred_proba.shape == (len(y_test), 2)
        assert np.allclose(pred_proba.sum(axis=1), 1.0)
    
    def test_data_consistency(self, sample_dataset):
        """Test data consistency throughout pipeline."""
        preprocessor = DataPreprocessor()
        
        # Load and process data
        raw_data = preprocessor.load_data(sample_dataset / "raw" / "creditcard.csv")
        processed_data = preprocessor.preprocess(raw_data)
        train, val, test = preprocessor.split_data(processed_data)
        
        # Check that all splits have the same columns
        assert set(train.columns) == set(val.columns) == set(test.columns)
        
        # Check that all splits have the target variable
        assert 'Class' in train.columns
        assert 'Class' in val.columns
        assert 'Class' in test.columns
        
        # Check that splits sum to total
        total_after_split = len(train) + len(val) + len(test)
        assert total_after_split == len(processed_data)
    
    def test_model_persistence(self, sample_dataset, temp_data_dir):
        """Test that model can be saved and loaded."""
        import joblib
        
        # Train a simple model
        preprocessor = DataPreprocessor()
        raw_data = preprocessor.load_data(sample_dataset / "raw" / "creditcard.csv")
        processed_data = preprocessor.preprocess(raw_data)
        train, val, test = preprocessor.split_data(processed_data)
        
        X_train = train.drop('Class', axis=1)
        y_train = train['Class']
        
        model_params = {
            'n_estimators': 10,
            'max_depth': 3,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        model = train_model(X_train, y_train, model_params)
        
        # Save model
        model_path = temp_data_dir / "model.pkl"
        joblib.dump(model, model_path)
        
        # Load model
        loaded_model = joblib.load(model_path)
        
        # Test that loaded model works
        X_val = val.drop('Class', axis=1)
        predictions = loaded_model.predict(X_val)
        assert len(predictions) == len(X_val)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])