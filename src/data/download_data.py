"""
Script to download the Credit Card Fraud Detection dataset.
"""
import os
import logging
import zipfile
from pathlib import Path
import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("data/raw")
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
# Alternative: You can use Kaggle API
# KAGGLE_DATASET = "mlg-ulb/creditcardfraud"


def download_from_url(url: str, output_path: Path) -> None:
    """Download file from URL."""
    logger.info(f"Downloading dataset from {url}")
    
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\rProgress: {progress:.1f}%", end='')
    
    print()  # New line after progress
    logger.info(f"Dataset downloaded to {output_path}")


def download_from_kaggle(dataset_name: str, output_dir: Path) -> None:
    """Download dataset from Kaggle using kaggle API."""
    try:
        import kaggle
        logger.info(f"Downloading dataset from Kaggle: {dataset_name}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(output_dir),
            unzip=True
        )
        
        logger.info(f"Dataset downloaded to {output_dir}")
    except ImportError:
        logger.error("Kaggle library not installed. Install with: pip install kaggle")
        logger.info("You can download manually from: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        raise


def main():
    """Main function to download dataset."""
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    output_file = DATA_DIR / "creditcard.csv"
    
    if output_file.exists():
        logger.info(f"Dataset already exists at {output_file}")
        return
    
    try:
        # Try downloading from direct URL first
        download_from_url(DATASET_URL, output_file)
    except Exception as e:
        logger.warning(f"Failed to download from URL: {e}")
        logger.info("Attempting to download from Kaggle...")
        
        try:
            # Try Kaggle API
            download_from_kaggle("mlg-ulb/creditcardfraud", DATA_DIR)
        except Exception as kaggle_error:
            logger.error(f"Failed to download from Kaggle: {kaggle_error}")
            logger.info("Please download the dataset manually from:")
            logger.info("https://www.kaggle.com/mlg-ulb/creditcardfraud")
            logger.info(f"And place it in: {DATA_DIR}")
            raise
    
    logger.info("Dataset download complete!")


if __name__ == "__main__":
    main()
