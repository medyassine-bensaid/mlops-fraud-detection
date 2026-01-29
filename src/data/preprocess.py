"""
Data preprocessing module for fraud detection.
"""
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class to handle data preprocessing."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load data from CSV file."""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data."""
        logger.info("Preprocessing data...")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
            df = df.dropna()
        
        # Scale Amount column (Time is already anonymized)
        if 'Amount' in df.columns:
            df['Amount_scaled'] = self.scaler.fit_transform(df[['Amount']])
            df = df.drop('Amount', axis=1)
        
        logger.info("Preprocessing complete")
        return df
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        logger.info("Splitting data...")
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['Class']
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val['Class']
        )
        
        logger.info(f"Train size: {len(train)} ({len(train)/len(df)*100:.1f}%)")
        logger.info(f"Validation size: {len(val)} ({len(val)/len(df)*100:.1f}%)")
        logger.info(f"Test size: {len(test)} ({len(test)/len(df)*100:.1f}%)")
        
        return train, val, test
    
    def save_data(
        self, 
        train: pd.DataFrame, 
        val: pd.DataFrame, 
        test: pd.DataFrame,
        output_dir: Path
    ) -> None:
        """Save processed data."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train.to_csv(output_dir / "train.csv", index=False)
        val.to_csv(output_dir / "val.csv", index=False)
        test.to_csv(output_dir / "test.csv", index=False)
        
        # Save scaler
        joblib.dump(self.scaler, output_dir / "scaler.pkl")
        
        logger.info(f"Data saved to {output_dir}")


def main():
    """Main preprocessing pipeline."""
    # Paths
    raw_data_path = Path("data/raw/creditcard.csv")
    processed_data_dir = Path("data/processed")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_data(raw_data_path)
    
    # Preprocess
    df_processed = preprocessor.preprocess(df)
    
    # Split data
    train, val, test = preprocessor.split_data(df_processed)
    
    # Save data
    preprocessor.save_data(train, val, test, processed_data_dir)
    
    logger.info("Preprocessing pipeline complete!")


if __name__ == "__main__":
    main()