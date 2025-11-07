#!/bin/bash

# Script to verify HTTPS setup

echo "### Checking nginx container status ..."
docker compose --profile production ps nginx

echo ""
echo "### Testing nginx configuration ..."
docker compose exec nginx nginx -t

echo ""
echo "### Checking if certificates exist ..."
docker compose exec nginx ls -la /etc/letsencrypt/live/dev-google-ai.mosida.com/ 2>/dev/null || echo "Certificates not found!"

echo ""
echo "### Checking if nginx is listening on ports 80 and 443 ..."
docker compose exec nginx netstat -tlnp 2>/dev/null | grep -E ':(80|443)' || docker compose exec nginx ss -tlnp 2>/dev/null | grep -E ':(80|443)'

echo ""
echo "### Checking nginx error logs (last 20 lines) ..."
docker compose logs nginx --tail 20 | grep -i error || echo "No errors found in recent logs"

echo ""
echo "### Reloading nginx configuration ..."
docker compose exec nginx nginx -s reload

echo ""
echo "### Verification complete!"
echo "### Try accessing: https://dev-google-ai.mosida.com"

