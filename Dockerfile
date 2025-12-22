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

# Create startup script to fix permissions (runs as root, then switches to appuser)
RUN echo '#!/bin/bash\n\
set -e\n\
# Fix permissions for mounted volumes (run as root before switching user)\n\
if [ "$(id -u)" = "0" ]; then\n\
    chmod -R 777 /app/uploads /app/outputs 2>/dev/null || true\n\
    chown -R appuser:appuser /app/uploads /app/outputs 2>/dev/null || true\n\
    # Switch to appuser and execute the command\n\
    exec gosu appuser "$@"\n\
else\n\
    # Already running as appuser\n\
    exec "$@"\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Install gosu for user switching
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# Don't switch to appuser here - start.sh will do it after fixing permissions
# This allows the container to start as root and fix mounted volume permissions

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application with permission fix
CMD ["/app/start.sh", "python", "app.py"]
