#!/bin/bash

# Script to fix Docker network issues
# Run this if you see "network not found" errors

echo "### Cleaning up Docker networks and containers ..."

# Stop all containers (ignore errors)
docker compose --profile production down 2>/dev/null || true
docker compose down 2>/dev/null || true

# Remove any containers with our names
for container in image-converter-app image-converter-nginx image-converter-certbot; do
  docker rm -f "$container" 2>/dev/null || true
done

# Remove the network by name
docker network rm googleaistudio-poc_image-converter-network 2>/dev/null || true
docker network rm image-converter-network 2>/dev/null || true

# Prune unused networks (this will remove orphaned networks)
echo "### Pruning unused networks ..."
docker network prune -f

# Prune unused volumes (optional, but helps clean up)
echo "### Cleaning up unused volumes ..."
docker volume prune -f

echo ""
echo "### Starting services with fresh networks ..."
docker compose --profile production up -d

echo ""
echo "### Verifying services are running ..."
sleep 3
docker compose --profile production ps

echo ""
if docker compose --profile production ps | grep -q "Up"; then
  echo "### Services are running! You can now run: ./init-letsencrypt.sh"
else
  echo "### Warning: Some services may not be running. Check logs:"
  echo "   docker compose --profile production logs"
fi

