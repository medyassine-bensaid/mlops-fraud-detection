#!/bin/bash

# DVC Setup Script for MLOps Fraud Detection Project

echo "=== DVC Setup Script ==="
echo ""

# Check if DVC is installed
if ! command -v dvc &> /dev/null; then
    echo "❌ DVC not installed"
    echo "Installing DVC..."
    pip install dvc dvc-s3
fi

echo "✅ DVC is installed"
echo ""

# Initialize DVC
echo "Initializing DVC..."
dvc init --force

# Configure S3 remote
echo "Configuring S3 remote storage..."
S3_BUCKET=${S3_BUCKET_NAME:-mlops-fraud-detection-data}
AWS_REGION=${AWS_REGION:-us-east-1}

dvc remote add -d -f s3storage s3://$S3_BUCKET/dvc-storage
dvc remote modify s3storage region $AWS_REGION

echo "✅ DVC remote configured: s3://$S3_BUCKET/dvc-storage"
echo ""

# Track data files (if they exist)
echo "Tracking data files with DVC..."

if [ -f "data/raw/creditcard.csv" ]; then
    echo "  - Tracking data/raw/creditcard.csv"
    dvc add data/raw/creditcard.csv
fi

if [ -f "data/processed/train.csv" ]; then
    echo "  - Tracking data/processed/train.csv"
    dvc add data/processed/train.csv
fi

if [ -f "data/processed/val.csv" ]; then
    echo "  - Tracking data/processed/val.csv"
    dvc add data/processed/val.csv
fi

if [ -f "data/processed/test.csv" ]; then
    echo "  - Tracking data/processed/test.csv"
    dvc add data/processed/test.csv
fi

echo ""
echo "=== DVC Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Commit .dvc files to Git:"
echo "   git add data/**/*.dvc .dvc/config"
echo "   git commit -m 'Setup DVC and track data'"
echo ""
echo "2. Push data to S3:"
echo "   dvc push"
echo ""
echo "3. Team members can pull data:"
echo "   dvc pull"
echo ""
echo "For more information, see DVC_GUIDE.md"