#!/bin/bash

# Direct fix for nginx.conf certificate path on server

echo "### Fixing nginx.conf certificate path ..."

# Check current content
echo "### Checking current certificate paths in nginx.conf ..."
grep -n "ssl_certificate" nginx.conf | head -2

# Create backup
cp nginx.conf nginx.conf.bak

# More robust replacement - handle both with and without trailing slashes
sed -i 's|/etc/letsencrypt/live/dev-google-ai\.mosida\.com/|/etc/letsencrypt/live/dev-google-ai.mosida.com-0001/|g' nginx.conf

# Verify the change
echo ""
echo "### Updated certificate paths:"
grep -n "ssl_certificate" nginx.conf | head -2

# Check if the change was made
if grep -q "dev-google-ai.mosida.com-0001" nginx.conf; then
    echo ""
    echo "### Certificate path updated successfully!"
    echo "### Testing configuration ..."
    
    # Test nginx config
    docker compose exec nginx nginx -t
    
    if [ $? -eq 0 ]; then
        echo "### Configuration test passed!"
        echo "### Reloading nginx ..."
        docker compose exec nginx nginx -s reload
        echo ""
        echo "✓ HTTPS should now work!"
        echo "✓ Visit: https://dev-google-ai.mosida.com"
        rm -f nginx.conf.bak
    else
        echo "### Error: Configuration test failed"
        echo "### Restoring backup ..."
        mv nginx.conf.bak nginx.conf
        echo ""
        echo "### Manual fix required. Please edit nginx.conf and change:"
        echo "   /etc/letsencrypt/live/dev-google-ai.mosida.com/"
        echo "   to:"
        echo "   /etc/letsencrypt/live/dev-google-ai.mosida.com-0001/"
        exit 1
    fi
else
    echo ""
    echo "### Error: Replacement didn't work. File may have different format."
    echo "### Restoring backup ..."
    mv nginx.conf.bak nginx.conf
    echo ""
    echo "### Please manually edit nginx.conf:"
    echo "   Find: ssl_certificate /etc/letsencrypt/live/dev-google-ai.mosida.com/fullchain.pem;"
    echo "   Replace with: ssl_certificate /etc/letsencrypt/live/dev-google-ai.mosida.com-0001/fullchain.pem;"
    echo ""
    echo "   Find: ssl_certificate_key /etc/letsencrypt/live/dev-google-ai.mosida.com/privkey.pem;"
    echo "   Replace with: ssl_certificate_key /etc/letsencrypt/live/dev-google-ai.mosida.com-0001/privkey.pem;"
    exit 1
fi
