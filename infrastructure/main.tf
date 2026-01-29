terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# S3 Bucket for data storage
resource "aws_s3_bucket" "data_bucket" {
  bucket = var.s3_bucket_name
  
  tags = {
    Name        = "MLOps Fraud Detection Data"
    Environment = var.environment
    Project     = "fraud-detection"
  }
}

resource "aws_s3_bucket_versioning" "data_bucket_versioning" {
  bucket = aws_s3_bucket.data_bucket.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_bucket_encryption" {
  bucket = aws_s3_bucket.data_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 Bucket for model artifacts
resource "aws_s3_bucket" "model_bucket" {
  bucket = "${var.s3_bucket_name}-models"
  
  tags = {
    Name        = "MLOps Model Artifacts"
    Environment = var.environment
    Project     = "fraud-detection"
  }
}

resource "aws_s3_bucket_versioning" "model_bucket_versioning" {
  bucket = aws_s3_bucket.model_bucket.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# ECR Repository for Docker images
resource "aws_ecr_repository" "fraud_detection" {
  name                 = "fraud-detection"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
  
  tags = {
    Name        = "Fraud Detection Repository"
    Environment = var.environment
    Project     = "fraud-detection"
  }
}


# Lambda Function
resource "aws_lambda_function" "fraud_detection" {
  function_name = "fraud-detection-api"
  role          = "arn:aws:sts::549009464329:assumed-role/voclabs/user4433534=mohamedyassine.bensaid2@gmail.com"
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.fraud_detection.repository_url}:latest"
  timeout       = 60
  memory_size   = 1024

  environment {
    variables = {
      MODEL_BUCKET = aws_s3_bucket.model_bucket.id
      DATA_BUCKET  = aws_s3_bucket.data_bucket.id
      ENVIRONMENT  = var.environment
    }
  }
  
  tags = {
    Name        = "Fraud Detection API"
    Environment = var.environment
    Project     = "fraud-detection"
  }
}

# API Gateway v2 (HTTP API)
resource "aws_apigatewayv2_api" "fraud_detection" {
  name          = "fraud-detection-api"
  protocol_type = "HTTP"
  
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["POST", "GET", "OPTIONS"]
    allow_headers = ["content-type", "authorization"]
  }
  
  tags = {
    Name        = "Fraud Detection API Gateway"
    Environment = var.environment
    Project     = "fraud-detection"
  }
}

# API Gateway Integration with Lambda
resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id           = aws_apigatewayv2_api.fraud_detection.id
  integration_type = "AWS_PROXY"
  integration_uri  = aws_lambda_function.fraud_detection.invoke_arn
  
  payload_format_version = "2.0"
}

# API Gateway Route for POST /predict
resource "aws_apigatewayv2_route" "predict" {
  api_id    = aws_apigatewayv2_api.fraud_detection.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

# API Gateway Route for GET /health
resource "aws_apigatewayv2_route" "health" {
  api_id    = aws_apigatewayv2_api.fraud_detection.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

# API Gateway Stage
resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.fraud_detection.id
  name        = "$default"
  auto_deploy = true
  
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway_logs.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
    })
  }
  
  tags = {
    Name        = "Default Stage"
    Environment = var.environment
    Project     = "fraud-detection"
  }
}

# Lambda Permission for API Gateway
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.fraud_detection.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.fraud_detection.execution_arn}/*/*"
}

# CloudWatch Log Group for Lambda
resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${aws_lambda_function.fraud_detection.function_name}"
  retention_in_days = 7
  
  tags = {
    Name        = "Lambda Logs"
    Environment = var.environment
    Project     = "fraud-detection"
  }
}

# CloudWatch Log Group for API Gateway
resource "aws_cloudwatch_log_group" "api_gateway_logs" {
  name              = "/aws/apigateway/fraud-detection-api"
  retention_in_days = 7
  
  tags = {
    Name        = "API Gateway Logs"
    Environment = var.environment
    Project     = "fraud-detection"
  }
}

# CloudWatch Alarm for Lambda Errors
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "fraud-detection-lambda-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = "60"
  statistic           = "Sum"
  threshold           = "5"
  alarm_description   = "This metric monitors lambda errors"
  treat_missing_data  = "notBreaching"
  
  dimensions = {
    FunctionName = aws_lambda_function.fraud_detection.function_name
  }
  
  tags = {
    Name        = "Lambda Error Alarm"
    Environment = var.environment
    Project     = "fraud-detection"
  }
}

# CloudWatch Alarm for Lambda Duration
resource "aws_cloudwatch_metric_alarm" "lambda_duration" {
  alarm_name          = "fraud-detection-lambda-duration"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = "60"
  statistic           = "Average"
  threshold           = "30000"  # 30 seconds
  alarm_description   = "This metric monitors lambda duration"
  treat_missing_data  = "notBreaching"
  
  dimensions = {
    FunctionName = aws_lambda_function.fraud_detection.function_name
  }
  
  tags = {
    Name        = "Lambda Duration Alarm"
    Environment = var.environment
    Project     = "fraud-detection"
  }
}

# SNS Topic for Alerts (Optional)
resource "aws_sns_topic" "alerts" {
  name = "fraud-detection-alerts"
  
  tags = {
    Name        = "Fraud Detection Alerts"
    Environment = var.environment
    Project     = "fraud-detection"
  }
}
