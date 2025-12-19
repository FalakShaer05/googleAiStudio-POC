# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production

# Install minimal system dependencies (only what's absolutely needed)
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY v2/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy v2 application code
COPY v2/ .

# Create necessary directories
RUN mkdir -p uploads outputs

# Create non-root user for security with specific UID/GID to match host
RUN groupadd -r -g 1000 appuser && useradd -r -u 1000 -g appuser appuser
RUN chown -R appuser:appuser /app

# Create startup script to fix permissions
RUN echo '#!/bin/bash\n\
# Fix permissions for mounted volumes\n\
chmod -R 755 /app/uploads /app/outputs 2>/dev/null || true\n\
chown -R appuser:appuser /app/uploads /app/outputs 2>/dev/null || true\n\
# Start the application\n\
exec "$@"' > /app/start.sh && chmod +x /app/start.sh

USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application with permission fix
CMD ["/app/start.sh", "python", "app.py"]
