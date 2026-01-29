# Terraform Variables Example
# Copy this file to terraform.tfvars and update with your values
# DO NOT commit terraform.tfvars to version control!

# AWS Region
aws_region = "us-east-1"

# S3 Bucket Name (MUST BE GLOBALLY UNIQUE)
# Change this to something unique, e.g., add your name or random string
s3_bucket_name = "mlops-fraud-detection-yourname-12345"

# Environment
environment = "production"

# Project Name
project_name = "fraud-detection"

# Lambda Configuration
lambda_timeout = 60
lambda_memory  = 1024

# Log Retention
log_retention_days = 7

# Additional Tags (optional)
tags = {
  Owner       = "Your Name"
  CostCenter  = "ML Team"
  ManagedBy   = "Terraform"
}
