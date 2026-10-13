FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configurations
COPY . .

# Generate data splits and train baseline models if not already present
RUN python -m simulation.generate_data && python -m src.models.train

EXPOSE 8000 8501
