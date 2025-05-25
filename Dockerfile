FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the application
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files for production (security best practice)
# Using explicit COPY commands instead of recursive copy to minimize security risk
COPY --chown=appuser:appuser yahoofinance/ ./yahoofinance/
COPY --chown=appuser:appuser trade.py ./
COPY --chown=appuser:appuser setup.py ./
COPY --chown=appuser:appuser pyproject.toml ./

# Copy essential input data files needed for application functionality
COPY --chown=appuser:appuser yahoofinance/input/ ./yahoofinance/input/

# Create directories for data with proper ownership and set secure permissions
RUN mkdir -p /app/logs && chown -R appuser:appuser /app/logs && chmod -R 755 /app/logs

# Set secure permissions on copied files - remove write permissions for group/others
# This addresses the SonarCloud security hotspot about write permissions on copied resources
RUN find /app -type f -exec chmod 644 {} \; && \
    find /app -type d -exec chmod 755 {} \; && \
    find /app -name "*.py" -exec chmod 644 {} \; && \
    find /app -name "*.sh" -exec chmod 755 {} \; && \
    chmod -R go-w /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Switch to non-root user
USER appuser

# Command to run when the container starts
CMD ["python", "trade.py"]