#!/bin/bash

echo "### Fixing HTTPS binding issue ..."

echo ""
echo "### Step 1: Checking certificate permissions ..."
docker compose exec nginx ls -la /etc/letsencrypt/live/dev-google-ai.mosida.com/

echo ""
echo "### Step 2: Testing nginx configuration ..."
docker compose exec nginx nginx -t

if [ $? -ne 0 ]; then
    echo "ERROR: Nginx configuration test failed!"
    exit 1
fi

echo ""
echo "### Step 3: Checking current nginx processes ..."
docker compose exec nginx ps aux | grep nginx

echo ""
echo "### Step 4: Stopping nginx container ..."
docker compose stop nginx

echo ""
echo "### Step 5: Starting nginx container fresh ..."
docker compose --profile production up -d nginx

echo ""
echo "### Step 6: Waiting for nginx to start ..."
sleep 5

echo ""
echo "### Step 7: Checking if nginx is listening on both ports ..."
docker compose exec nginx sh -c "netstat -tlnp 2>/dev/null || ss -tlnp" | grep -E ":(80|443)"

echo ""
echo "### Step 8: Checking nginx error logs ..."
docker compose logs nginx --tail 20 | grep -i -E "(error|warn|ssl|443)" || echo "No SSL-related errors found"

echo ""
echo "### Step 9: Verifying HTTPS server block is active ..."
if docker compose exec nginx nginx -T 2>/dev/null | grep -q "listen 443"; then
    echo "✓ HTTPS server block is in active configuration"
else
    echo "✗ HTTPS server block NOT found in active configuration!"
    echo "The nginx.conf file might not be mounted correctly."
fi

echo ""
echo "### Fix complete!"
echo "### If port 443 is still not listening, check:"
echo "   1. docker compose logs nginx"
echo "   2. Verify nginx.conf is mounted: docker compose exec nginx cat /etc/nginx/nginx.conf | grep 'listen 443'"

