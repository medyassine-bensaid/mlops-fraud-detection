"""
Script to upload data to S3.
"""
import os
import logging
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class S3Uploader:
    """Class to handle S3 uploads."""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=region)
        
    def upload_file(self, file_path: Path, s3_key: str) -> bool:
        """Upload a file to S3."""
        try:
            logger.info(f"Uploading {file_path} to s3://{self.bucket_name}/{s3_key}")
            self.s3_client.upload_file(
                str(file_path), 
                self.bucket_name, 
                s3_key
            )
            logger.info(f"Successfully uploaded {file_path}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload {file_path}: {e}")
            return False
    
    def upload_directory(self, directory: Path, s3_prefix: str = "") -> None:
        """Upload all files in a directory to S3."""
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(directory)
                s3_key = f"{s3_prefix}/{relative_path}" if s3_prefix else str(relative_path)
                self.upload_file(file_path, s3_key)


def main():
    """Main function to upload data to S3."""
    # Get bucket name from environment
    bucket_name = os.getenv("S3_BUCKET_NAME", "mlops-fraud-detection-data")
    region = os.getenv("AWS_REGION", "us-east-1")
    
    # Initialize uploader
    uploader = S3Uploader(bucket_name, region)
    
    # Upload processed data
    processed_data_dir = Path("data/processed")
    if processed_data_dir.exists():
        logger.info("Uploading processed data...")
        uploader.upload_directory(processed_data_dir, "data/processed")
    
    # Upload raw data
    raw_data_dir = Path("data/raw")
    if raw_data_dir.exists():
        logger.info("Uploading raw data...")
        uploader.upload_directory(raw_data_dir, "data/raw")
    
    logger.info("Upload complete!")


if __name__ == "__main__":
    main()