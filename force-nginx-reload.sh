#!/bin/bash

# Force nginx container to see the updated nginx.conf

echo "### Forcing nginx to reload with updated config ..."

# Verify the host file is correct
echo "### Host nginx.conf certificate paths:"
grep "ssl_certificate" nginx.conf | grep -v "^#"

echo ""
echo "### Stopping nginx container ..."
docker compose stop nginx

echo ""
echo "### Removing nginx container to force fresh mount ..."
docker compose rm -f nginx

echo ""
echo "### Starting nginx with fresh volume mount ..."
docker compose --profile production up -d nginx

echo ""
echo "### Waiting for nginx to start ..."
sleep 5

echo ""
echo "### Verifying what nginx container sees now ..."
docker compose exec nginx cat /etc/nginx/nginx.conf | grep "ssl_certificate" | grep -v "^#"

echo ""
echo "### Testing nginx configuration ..."
docker compose exec nginx nginx -t

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Nginx configuration is valid!"
    echo "✓ HTTPS should now work: https://dev-google-ai.mosida.com"
else
    echo ""
    echo "### Error: Configuration still invalid"
    echo "### The volume mount may not be working. Check docker-compose.yml"
fi

