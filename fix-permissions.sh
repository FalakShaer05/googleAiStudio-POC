#!/bin/bash

# Fix permissions script for Docker container
echo "🔧 Fixing permissions for Docker container..."

# Stop and remove existing containers
echo "📦 Stopping existing containers..."
docker-compose down

# Remove existing images to force rebuild
echo "🗑️  Removing existing images..."
docker-compose build --no-cache

# Fix host directory permissions
echo "🔐 Setting correct permissions on host directories..."
sudo chmod -R 755 uploads outputs
sudo chown -R 1000:1000 uploads outputs

# Start the containers
echo "🚀 Starting containers with fixed permissions..."
docker-compose up -d

echo "✅ Done! The permission issue should now be resolved."
echo "📝 If you still have issues, you may need to run:"
echo "   sudo chown -R 1000:1000 uploads outputs"
