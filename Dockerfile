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

# Copy requirements first for better caching (root owns, read-only for others)
COPY --chmod=444 requirements.txt .

# Install Python dependencies as root (for system-wide installation)
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -f requirements.txt

# Copy only necessary application code with read-only permissions
# Files are owned by root but readable by appuser
COPY --chmod=444 trade.py trade_test.py ./
COPY --chmod=555 trade_modules ./trade_modules
COPY --chmod=555 yahoofinance ./yahoofinance

# Fix Python module permissions (directories need execute for traversal)
RUN find /app -type d -exec chmod 555 {} \; && \
    find /app -type f -name "*.py" -exec chmod 444 {} \;

# Create necessary runtime directories with proper ownership
RUN mkdir -p /app/logs /app/yahoofinance/input /app/yahoofinance/output /app/config && \
    chown -R appuser:appuser /app/logs /app/yahoofinance/input /app/yahoofinance/output /app/config && \
    chmod 755 /app/logs /app/yahoofinance/input /app/yahoofinance/output /app/config

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

# Security notes:
# - Application code is read-only (444 for files, 555 for directories)
# - Config must be mounted from outside: -v $(pwd)/config.yaml:/app/config/config.yaml:ro
# - Data directories are writable only by appuser
# - No sensitive data is included in the image

# Example usage:
# docker build -t etoro-analysis .
# docker run -v $(pwd)/config.yaml:/app/config/config.yaml:ro -v $(pwd)/data:/app/yahoofinance/input etoro-analysis python trade.py -o p -t e