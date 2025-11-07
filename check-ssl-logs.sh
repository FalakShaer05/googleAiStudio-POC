#!/bin/bash

echo "### Checking nginx error logs for SSL/443 issues ..."
docker compose logs nginx 2>&1 | grep -i -E "(ssl|443|certificate|error|warn)" | tail -30

echo ""
echo "### Checking if nginx process can see port 443 binding ..."
docker compose exec nginx sh -c "netstat -tlnp 2>/dev/null || ss -tlnp" | grep 443

echo ""
echo "### Checking nginx configuration for HTTPS server block ..."
docker compose exec nginx nginx -T 2>/dev/null | grep -A 5 "listen 443" || echo "HTTPS server block not found in active config!"

echo ""
echo "### Attempting to restart nginx container ..."
docker compose restart nginx
sleep 3

echo ""
echo "### Checking ports after restart ..."
docker compose exec nginx sh -c "netstat -tlnp 2>/dev/null || ss -tlnp" | grep -E ":(80|443)"

