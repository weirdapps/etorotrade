# Simple Docker container for eToro Trading Analysis
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies with explicit --no-install-recommends
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for running the application
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements first for better caching
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary application files (exclude sensitive data)
# Use .dockerignore to control what gets copied
COPY --chown=appuser:appuser trade.py trade_test.py config.yaml ./
COPY --chown=appuser:appuser trade_modules ./trade_modules
COPY --chown=appuser:appuser yahoofinance ./yahoofinance
COPY --chown=appuser:appuser examples ./examples

# Create necessary directories with proper ownership
RUN mkdir -p logs yahoofinance/input yahoofinance/output && \
    chown -R appuser:appuser logs yahoofinance/input yahoofinance/output

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Default command
CMD ["python", "trade.py", "--help"]

# Example usage:
# docker build -t etoro-analysis .
# docker run -v $(pwd)/data:/app/yahoofinance/input etoro-analysis python trade.py -o p -t e