# Agentic Data Science Advisor - Docker Image
FROM python:3.12-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY tests/ ./tests/

# Create reports directory
RUN mkdir -p reports

# Default command runs tests to verify installation
CMD ["python", "-m", "pytest", "tests/", "-v"]