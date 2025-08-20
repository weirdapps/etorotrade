# Simple Docker container for eToro Trading Analysis
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs yahoofinance/input yahoofinance/output

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "trade.py", "--help"]

# Example usage:
# docker build -t etoro-analysis .
# docker run -v $(pwd)/data:/app/yahoofinance/input etoro-analysis python trade.py -o p -t e