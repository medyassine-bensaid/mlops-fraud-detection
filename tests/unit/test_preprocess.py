"""
Unit tests for data preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.preprocess import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Time': np.random.rand(n_samples) * 172800,
        'Amount': np.random.rand(n_samples) * 1000,
        'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    }
    
    # Add V1-V28 features
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    """Create preprocessor instance."""
    return DataPreprocessor()


class TestDataPreprocessor:
    """Test DataPreprocessor class."""
    
    def test_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor is not None
        assert hasattr(preprocessor, 'scaler')
    
    def test_load_data_shape(self, sample_data, tmp_path):
        """Test data loading."""
        # Save sample data
        data_file = tmp_path / "test_data.csv"
        sample_data.to_csv(data_file, index=False)
        
        # Load data
        preprocessor = DataPreprocessor()
        loaded_data = preprocessor.load_data(data_file)
        
        assert loaded_data.shape == sample_data.shape
        assert list(loaded_data.columns) == list(sample_data.columns)
    
    def test_preprocess_scales_amount(self, preprocessor, sample_data):
        """Test that preprocessing scales the Amount column."""
        processed_data = preprocessor.preprocess(sample_data)
        
        # Check that Amount is removed and Amount_scaled is added
        assert 'Amount' not in processed_data.columns
        assert 'Amount_scaled' in processed_data.columns
        
        # Check that Amount_scaled is standardized (mean ~0, std ~1)
        assert abs(processed_data['Amount_scaled'].mean()) < 0.1
        assert abs(processed_data['Amount_scaled'].std() - 1.0) < 0.1
    
    def test_preprocess_handles_missing_values(self, preprocessor, sample_data):
        """Test that preprocessing handles missing values."""
        # Add missing values
        sample_data_with_na = sample_data.copy()
        sample_data_with_na.loc[0:10, 'V1'] = np.nan
        
        processed_data = preprocessor.preprocess(sample_data_with_na)
        
        # Check that missing values are removed
        assert processed_data.isnull().sum().sum() == 0
        assert len(processed_data) < len(sample_data_with_na)
    
    def test_split_data_sizes(self, preprocessor, sample_data):
        """Test data splitting."""
        processed_data = preprocessor.preprocess(sample_data)
        train, val, test = preprocessor.split_data(
            processed_data,
            test_size=0.2,
            val_size=0.1
        )
        
        # Check sizes
        total_size = len(processed_data)
        assert len(test) == pytest.approx(total_size * 0.2, rel=0.01)
        assert len(val) == pytest.approx(total_size * 0.1, rel=0.01)
        assert len(train) == pytest.approx(total_size * 0.7, rel=0.01)
        
        # Check no overlap
        assert len(set(train.index) & set(val.index)) == 0
        assert len(set(train.index) & set(test.index)) == 0
        assert len(set(val.index) & set(test.index)) == 0
    
    def test_split_data_stratification(self, preprocessor, sample_data):
        """Test that data splitting maintains class balance."""
        processed_data = preprocessor.preprocess(sample_data)
        train, val, test = preprocessor.split_data(processed_data)
        
        # Check class distribution is similar across splits
        train_fraud_ratio = train['Class'].mean()
        val_fraud_ratio = val['Class'].mean()
        test_fraud_ratio = test['Class'].mean()
        
        # All splits should have similar fraud ratios
        assert abs(train_fraud_ratio - val_fraud_ratio) < 0.01
        assert abs(train_fraud_ratio - test_fraud_ratio) < 0.01
    
    def test_save_data(self, preprocessor, sample_data, tmp_path):
        """Test data saving."""
        processed_data = preprocessor.preprocess(sample_data)
        train, val, test = preprocessor.split_data(processed_data)
        
        output_dir = tmp_path / "processed"
        preprocessor.save_data(train, val, test, output_dir)
        
        # Check files exist
        assert (output_dir / "train.csv").exists()
        assert (output_dir / "val.csv").exists()
        assert (output_dir / "test.csv").exists()
        assert (output_dir / "scaler.pkl").exists()
        
        # Check files can be loaded
        loaded_train = pd.read_csv(output_dir / "train.csv")
        assert len(loaded_train) == len(train)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])