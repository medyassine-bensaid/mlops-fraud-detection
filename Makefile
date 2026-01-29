.PHONY: setup download-data train test lint format deploy monitor clean help

# Variables
PYTHON := python
PIP := pip3
PYTEST := pytest
BLACK := black
PYLINT := pylint
DOCKER := docker
AWS := aws

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

setup: ## Setup development environment
	@echo "Setting up environment..."
	$(PYTHON) -m venv venv
	. venv/bin/activate && $(PIP) install --upgrade pip
	. venv/bin/activate && $(PIP) install -r requirements.txt
	. venv/bin/activate && pre-commit install
	@echo "✓ Environment setup complete"

download-data: ## Download dataset
	@echo "Downloading dataset..."
	$(PYTHON) src/data/download_data.py
	@echo "✓ Dataset downloaded"

preprocess: ## Preprocess data
	@echo "Preprocessing data..."
	$(PYTHON) src/data/preprocess.py
	@echo "✓ Data preprocessed"

upload-data: ## Upload data to S3
	@echo "Uploading data to S3..."
	$(PYTHON) src/data/upload_to_s3.py
	@echo "✓ Data uploaded"

train: ## Train model
	@echo "Training model..."
	$(PYTHON) src/models/train.py
	@echo "✓ Model trained"

evaluate: ## Evaluate model
	@echo "Evaluating model..."
	$(PYTHON) src/models/evaluate.py
	@echo "✓ Model evaluated"

test: ## Run all tests
	@echo "Running tests..."
	$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "✓ Tests complete"

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	$(PYTEST) tests/unit/ -v
	@echo "✓ Unit tests complete"

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	$(PYTEST) tests/integration/ -v
	@echo "✓ Integration tests complete"

lint: ## Run linting
	@echo "Running linters..."
	$(PYLINT) src/
	flake8 src/
	@echo "✓ Linting complete"

format: ## Format code
	@echo "Formatting code..."
	$(BLACK) src/ tests/
	isort src/ tests/
	@echo "✓ Code formatted"

type-check: ## Run type checking
	@echo "Running type checks..."
	mypy src/
	@echo "✓ Type checking complete"

docker-build: ## Build Docker image
	@echo "Building Docker image..."
	$(DOCKER) build -t fraud-detection:latest -f src/deployment/Dockerfile .
	@echo "✓ Docker image built"

docker-run: ## Run Docker container locally
	@echo "Running Docker container..."
	$(DOCKER) run -p 8080:8080 fraud-detection:latest

docker-push: ## Push Docker image to ECR
	@echo "Pushing Docker image to ECR..."
	$(AWS) ecr get-login-password --region $(AWS_REGION) | $(DOCKER) login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
	$(DOCKER) tag fraud-detection:latest $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/fraud-detection:latest
	$(DOCKER) push $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/fraud-detection:latest
	@echo "✓ Docker image pushed"

terraform-init: ## Initialize Terraform
	@echo "Initializing Terraform..."
	cd infrastructure && terraform init
	@echo "✓ Terraform initialized"

terraform-plan: ## Plan Terraform changes
	@echo "Planning Terraform changes..."
	cd infrastructure && terraform plan
	@echo "✓ Terraform plan complete"

terraform-apply: ## Apply Terraform changes
	@echo "Applying Terraform changes..."
	cd infrastructure && terraform apply -auto-approve
	@echo "✓ Infrastructure provisioned"

terraform-destroy: ## Destroy Terraform resources
	@echo "Destroying Terraform resources..."
	cd infrastructure && terraform destroy -auto-approve
	@echo "✓ Infrastructure destroyed"

deploy: docker-build docker-push ## Deploy model to AWS
	@echo "Deploying model..."
	$(PYTHON) src/deployment/deploy.py
	@echo "✓ Model deployed"

monitor: ## Generate monitoring report
	@echo "Generating monitoring report..."
	$(PYTHON) src/monitoring/monitor.py
	@echo "✓ Monitoring report generated"

dashboard: ## Start monitoring dashboard
	@echo "Starting monitoring dashboard..."
	$(PYTHON) src/monitoring/dashboard.py

prefect-start: ## Start Prefect server
	@echo "Starting Prefect server..."
	prefect server start

prefect-deploy: ## Deploy Prefect flows
	@echo "Deploying Prefect flows..."
	prefect deploy --all
	@echo "✓ Prefect flows deployed"

mlflow-start: ## Start MLflow server
	@echo "Starting MLflow server..."
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000


dvc-setup: ## Setup DVC for data versioning
	@echo "Setting up DVC..."
	./setup_dvc.sh
	@echo "✓ DVC setup complete"

dvc-add: ## Add data files to DVC tracking
	@echo "Adding data to DVC..."
	dvc add data/raw/creditcard.csv || true
	dvc add data/processed/train.csv || true
	dvc add data/processed/val.csv || true
	dvc add data/processed/test.csv || true
	@echo "✓ Data added to DVC"

dvc-push: ## Push data to remote storage
	@echo "Pushing data to S3..."
	dvc push
	@echo "✓ Data pushed to remote"

dvc-pull: ## Pull data from remote storage
	@echo "Pulling data from S3..."
	dvc pull
	@echo "✓ Data pulled from remote"

dvc-status: ## Show DVC status
	@echo "DVC status:"
	dvc status

clean: ## Clean generated files
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	@echo "✓ Cleanup complete"

all: setup download-data preprocess train test lint ## Run complete pipeline
	@echo "✓ Complete pipeline executed"