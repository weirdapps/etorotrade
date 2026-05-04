FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the application
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install Poetry, EXPORT pinned/hashed production deps to a temp requirements
# file, then `pip install --only-binary :all: --require-hashes` (avoids running
# arbitrary setup.py from source distributions). Poetry is removed afterwards.
COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry==2.4.0 poetry-plugin-export==1.10.0 && \
    poetry export --only main -f requirements.txt -o /tmp/req.txt && \
    pip install --no-cache-dir --only-binary :all: --require-hashes -r /tmp/req.txt && \
    pip uninstall -y poetry poetry-plugin-export && \
    rm -f /tmp/req.txt

# Copy only necessary files for production (security best practice)
# Using explicit COPY commands instead of recursive copy to minimize security risk
COPY yahoofinance/ ./yahoofinance/
COPY trade.py ./

# Copy essential input data files needed for application functionality
COPY yahoofinance/input/ ./yahoofinance/input/

# Set proper ownership and secure permissions on all copied files
# This addresses the SonarCloud security hotspot about write permissions on copied resources
RUN chown -R appuser:appuser /app && \
    find /app -type f -exec chmod 644 {} \; && \
    find /app -type d -exec chmod 755 {} \; && \
    find /app -name "*.py" -exec chmod 644 {} \; && \
    find /app -name "*.sh" -exec chmod 755 {} \; && \
    chmod -R go-w /app && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app/logs && \
    chmod -R 755 /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Switch to non-root user
USER appuser

# Command to run when the container starts
CMD ["python", "trade.py"]
