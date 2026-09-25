# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    HOME=/app \
    U2NET_HOME=/app/.u2net \
    REMBG_MODEL_ORDER=u2net \
    REMBG_MAX_SIDE=2048 \
    SKIP_FREEPIK=false \
    GEMINI_IMAGE_MODEL=gemini-3-pro-image \
    GEMINI_FALLBACK_IMAGE_MODEL=gemini-2.5-flash-image \
    GEMINI_MAX_RETRIES=3 \
    GEMINI_RETRY_INITIAL_DELAY_S=1.0 \
    GEMINI_RETRY_MAX_DELAY_S=10.0

# Runtime libs only (Pillow/rembg ship wheels — no -dev packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    fonts-dejavu-core \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY v2/requirements.txt .

# Install Python dependencies (no pip self-upgrade — flaky on slow networks)
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_RETRIES=15
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Bake rembg u2net weights so first Me Remix / Selfie Becoming request is not a cold download
RUN mkdir -p /app/.u2net && \
    python -c "from rembg import new_session; new_session('u2net')"

# Copy v2 application code
COPY v2/ .

# Create necessary directories
RUN mkdir -p uploads outputs

# Create non-root user for security with specific UID/GID to match host.
# Real home dir: rembg/pooch downloads models to U2NET_HOME (/app/.u2net).
RUN groupadd -r -g 1000 appuser && useradd -m -u 1000 -g appuser -d /home/appuser appuser
RUN chown -R appuser:appuser /app /home/appuser

# Create startup script to fix permissions (runs as root, then switches to appuser)
RUN echo '#!/bin/bash\n\
set -e\n\
# Fix permissions for mounted volumes (run as root before switching user)\n\
if [ "$(id -u)" = "0" ]; then\n\
    chmod -R 777 /app/uploads /app/outputs 2>/dev/null || true\n\
    chown -R appuser:appuser /app/uploads /app/outputs /app/.u2net 2>/dev/null || true\n\
    # Switch to appuser and execute the command\n\
    exec gosu appuser "$@"\n\
else\n\
    # Already running as appuser\n\
    exec "$@"\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Install gosu from GitHub release (avoids a second apt-get after COPY)
ARG GOSU_VERSION=1.17
RUN curl -fsSL -o /usr/local/bin/gosu \
      "https://github.com/tianon/gosu/releases/download/${GOSU_VERSION}/gosu-amd64" \
    && chmod +x /usr/local/bin/gosu \
    && gosu --version

# Don't switch to appuser here - start.sh will do it after fixing permissions
# This allows the container to start as root and fix mounted volume permissions

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application with permission fix
CMD ["/app/start.sh", "python", "app.py"]
