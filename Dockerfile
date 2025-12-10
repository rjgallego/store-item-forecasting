# Dockerfile

FROM python:3.11-slim

# Install system packages needed for some Python libs (e.g., pyarrow, catboost)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency list first (better for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default command: show help for the CLI
CMD ["python", "-m", "src.models.predict_catboost", "--help"]
