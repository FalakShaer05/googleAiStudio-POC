#!/bin/bash

# Direct fix for nginx.conf certificate path on server

echo "### Fixing nginx.conf certificate path ..."

# Update nginx.conf to use the correct certificate path
sed -i.bak 's|/etc/letsencrypt/live/dev-google-ai.mosida.com/|/etc/letsencrypt/live/dev-google-ai.mosida.com-0001/|g' nginx.conf

echo "### Updated nginx.conf"
echo "### Testing configuration ..."

# Test nginx config
docker compose exec nginx nginx -t

if [ $? -eq 0 ]; then
    echo "### Configuration test passed!"
    echo "### Reloading nginx ..."
    docker compose exec nginx nginx -s reload
    echo "✓ HTTPS should now work!"
    echo "✓ Visit: https://dev-google-ai.mosida.com"
else
    echo "### Error: Configuration test failed"
    echo "### Restoring backup ..."
    mv nginx.conf.bak nginx.conf
    echo "### Please check nginx.conf manually"
    exit 1
fi

# Remove backup if successful
rm -f nginx.conf.bak

echo ""
echo "### Done! Your site should now be secure."

