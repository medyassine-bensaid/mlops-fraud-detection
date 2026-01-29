output "s3_data_bucket_name" {
  description = "Name of the S3 data bucket"
  value       = aws_s3_bucket.data_bucket.id
}

output "s3_data_bucket_arn" {
  description = "ARN of the S3 data bucket"
  value       = aws_s3_bucket.data_bucket.arn
}

output "s3_model_bucket_name" {
  description = "Name of the S3 model bucket"
  value       = aws_s3_bucket.model_bucket.id
}

output "s3_model_bucket_arn" {
  description = "ARN of the S3 model bucket"
  value       = aws_s3_bucket.model_bucket.arn
}

output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.fraud_detection.repository_url
}

output "ecr_repository_arn" {
  description = "ARN of the ECR repository"
  value       = aws_ecr_repository.fraud_detection.arn
}

output "lambda_function_name" {
  description = "Name of the Lambda function"
  value       = aws_lambda_function.fraud_detection.function_name
}

output "lambda_function_arn" {
  description = "ARN of the Lambda function"
  value       = aws_lambda_function.fraud_detection.arn
}

output "lambda_function_invoke_arn" {
  description = "Invoke ARN of the Lambda function"
  value       = aws_lambda_function.fraud_detection.invoke_arn
}


output "lambda_role_arn" {
  value = var.existing_lambda_role_arn
  description = "ARN of the IAM role used by the Lambda function"
}


output "api_gateway_id" {
  description = "ID of the API Gateway"
  value       = aws_apigatewayv2_api.fraud_detection.id
}

output "api_gateway_endpoint" {
  description = "Endpoint URL of the API Gateway"
  value       = aws_apigatewayv2_stage.default.invoke_url
}

output "api_gateway_execution_arn" {
  description = "Execution ARN of the API Gateway"
  value       = aws_apigatewayv2_api.fraud_detection.execution_arn
}

output "cloudwatch_log_group_lambda" {
  description = "Name of the Lambda CloudWatch log group"
  value       = aws_cloudwatch_log_group.lambda_logs.name
}

output "cloudwatch_log_group_api_gateway" {
  description = "Name of the API Gateway CloudWatch log group"
  value       = aws_cloudwatch_log_group.api_gateway_logs.name
}

output "sns_topic_arn" {
  description = "ARN of the SNS topic for alerts"
  value       = aws_sns_topic.alerts.arn
}

output "region" {
  description = "AWS region"
  value       = var.aws_region
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}

# Useful outputs for deployment scripts
output "deployment_info" {
  description = "Deployment information"
  value = {
    api_endpoint        = aws_apigatewayv2_stage.default.invoke_url
    lambda_function     = aws_lambda_function.fraud_detection.function_name
    ecr_repository      = aws_ecr_repository.fraud_detection.repository_url
    data_bucket         = aws_s3_bucket.data_bucket.id
    model_bucket        = aws_s3_bucket.model_bucket.id
    region              = var.aws_region
  }
}

# Test commands
output "test_commands" {
  description = "Commands to test the deployment"
  value = <<-EOT
    # Test the API endpoint:
    curl -X POST ${aws_apigatewayv2_stage.default.invoke_url}/predict \
      -H "Content-Type: application/json" \
      -d '{"Time": 0, "V1": -1.36, "Amount": 149.62}'
    
    # Check Lambda logs:
    aws logs tail ${aws_cloudwatch_log_group.lambda_logs.name} --follow
    
    # Push Docker image to ECR:
    aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${aws_ecr_repository.fraud_detection.repository_url}
    docker tag fraud-detection:latest ${aws_ecr_repository.fraud_detection.repository_url}:latest
    docker push ${aws_ecr_repository.fraud_detection.repository_url}:latest
  EOT
}
