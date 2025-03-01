FROM python:3.13-alpine

WORKDIR /app

# Install build dependencies first
RUN apk add --no-cache \
  build-base \
  gcc \
  musl-dev \
  python3-dev \
  openblas-dev \
  lapack-dev \
  gfortran

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY paperless_ml_integration.py .

# Create directory for models
RUN mkdir -p /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "paperless_ml_integration.py"]