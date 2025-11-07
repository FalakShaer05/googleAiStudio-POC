#!/bin/bash

# Script to verify nginx config and reload

echo "### Verifying nginx.conf certificate paths ..."
grep "ssl_certificate" nginx.conf | grep -v "^#"

echo ""
echo "### Checking what nginx container sees ..."
docker compose exec nginx cat /etc/nginx/nginx.conf | grep "ssl_certificate" | grep -v "^#"

echo ""
echo "### Testing nginx configuration ..."
docker compose exec nginx nginx -t

if [ $? -eq 0 ]; then
    echo ""
    echo "### Configuration test passed! Reloading nginx ..."
    docker compose exec nginx nginx -s reload
    
    echo ""
    echo "### Checking if nginx reloaded successfully ..."
    sleep 2
    docker compose exec nginx nginx -t
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Nginx reloaded successfully!"
        echo "✓ Your site should now be secure: https://dev-google-ai.mosida.com"
    else
        echo ""
        echo "⚠️  Warning: nginx test failed after reload"
        echo "   Try restarting the container: docker compose restart nginx"
    fi
else
    echo ""
    echo "### Error: Configuration test failed"
    echo "### The nginx.conf file in the container may be different"
    echo "### Try restarting nginx container: docker compose restart nginx"
    exit 1
fi

