#!/bin/bash

# Script to fix certificate path mismatch
# This happens when Let's Encrypt creates a numbered certificate

DOMAIN="dev-google-ai.mosida.com"

echo "### Checking certificate paths ..."

# Find the actual certificate path
ACTUAL_PATH=$(docker compose run --rm --entrypoint "certbot certificates" certbot 2>/dev/null | grep -A 5 "$DOMAIN" | grep "Certificate Path" | awk '{print $3}' | sed 's|/fullchain.pem||' | sed 's|/etc/letsencrypt/live/||')

if [ -z "$ACTUAL_PATH" ]; then
    echo "Error: Could not find certificate path"
    exit 1
fi

echo "### Found certificate at: $ACTUAL_PATH"

# Check if it's the numbered version
if [[ "$ACTUAL_PATH" == *"-0001"* ]] || [[ "$ACTUAL_PATH" != "$DOMAIN" ]]; then
    echo "### Certificate path mismatch detected!"
    echo "### Actual path: $ACTUAL_PATH"
    echo "### Expected path: $DOMAIN"
    echo ""
    echo "### Option 1: Update nginx.conf to use the actual path"
    echo "### Option 2: Remove old certificate and get a fresh one with correct name"
    echo ""
    read -p "Update nginx.conf automatically? (Y/n) " choice
    
    if [[ "$choice" != "n" && "$choice" != "N" ]]; then
        echo "### Updating nginx.conf ..."
        
        # Update nginx.conf
        sed -i.bak "s|/etc/letsencrypt/live/$DOMAIN/|/etc/letsencrypt/live/$ACTUAL_PATH/|g" nginx.conf
        
        echo "### Testing nginx configuration ..."
        docker compose exec nginx nginx -t
        
        if [ $? -eq 0 ]; then
            echo "### Reloading nginx ..."
            docker compose exec nginx nginx -s reload
            echo "✓ nginx.conf updated and reloaded!"
        else
            echo "Error: nginx configuration test failed"
            mv nginx.conf.bak nginx.conf
            exit 1
        fi
        
        rm -f nginx.conf.bak
    else
        echo "### To fix manually, update nginx.conf:"
        echo "   Change: /etc/letsencrypt/live/$DOMAIN/"
        echo "   To:     /etc/letsencrypt/live/$ACTUAL_PATH/"
    fi
else
    echo "✓ Certificate path is correct: $ACTUAL_PATH"
fi

