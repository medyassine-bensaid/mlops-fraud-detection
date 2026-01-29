FROM public.ecr.aws/lambda/python:3.9

# Copy requirements
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install dependencies
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy application code
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY models/ ${LAMBDA_TASK_ROOT}/models/

# Set the CMD to your handler
CMD ["src.deployment.lambda_function.handler"]