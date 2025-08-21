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

# Install Python dependencies as root (for system-wide installation)
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary application code (no config files)
# Config should be provided via environment variables or mounted volumes
COPY --chown=appuser:appuser trade.py trade_test.py ./
COPY --chown=appuser:appuser trade_modules ./trade_modules
COPY --chown=appuser:appuser yahoofinance ./yahoofinance

# Create necessary directories with proper ownership
RUN mkdir -p logs yahoofinance/input yahoofinance/output /app/config && \
    chown -R appuser:appuser logs yahoofinance/input yahoofinance/output /app/config

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "trade.py", "--help"]

# Example usage:
# docker build -t etoro-analysis .
# docker run -v $(pwd)/config.yaml:/app/config.yaml:ro -v $(pwd)/data:/app/yahoofinance/input etoro-analysis python trade.py -o p -t e
# Note: Mount config.yaml as read-only (:ro) for security