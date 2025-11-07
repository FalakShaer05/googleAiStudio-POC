#!/bin/bash

# Script to get production SSL certificate
# This removes the staging certificate and gets a real one

if ! docker compose version > /dev/null 2>&1; then
  echo 'Error: docker compose is not installed or not available.' >&2
  exit 1
fi

DOMAIN="dev-google-ai.mosida.com"
EMAIL="falak@livao.com"

echo "### Getting PRODUCTION SSL certificate for $DOMAIN ..."
echo "### This will replace any existing staging certificate"
echo ""

# Check if certificates exist
if docker compose exec nginx test -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" 2>/dev/null; then
  echo "### Removing existing certificates ..."
  docker compose run --rm --entrypoint "\
    rm -Rf /etc/letsencrypt/live/$DOMAIN && \
    rm -Rf /etc/letsencrypt/archive/$DOMAIN && \
    rm -Rf /etc/letsencrypt/renewal/$DOMAIN.conf" certbot
  echo ""
fi

echo "### Requesting PRODUCTION Let's Encrypt certificate (STAGING=0) ..."
docker compose run --rm --entrypoint "\
  certbot certonly --webroot -w /var/www/certbot \
    --email $EMAIL \
    -d $DOMAIN \
    --rsa-key-size 4096 \
    --agree-tos \
    --force-renewal" certbot

if [ $? -eq 0 ]; then
  echo ""
  echo "### Production certificate obtained successfully!"
  echo "### Reloading nginx ..."
  docker compose exec nginx nginx -s reload
  
  echo ""
  echo "### Verifying certificate ..."
  sleep 2
  CERT_INFO=$(docker compose run --rm --entrypoint "certbot certificates" certbot 2>/dev/null | grep -A 10 "$DOMAIN")
  if echo "$CERT_INFO" | grep -qi "staging\|fake"; then
    echo "⚠️  WARNING: Still showing staging certificate!"
    echo "   You may need to wait a moment and reload nginx again"
  else
    echo "✓ Production certificate verified!"
    echo "✓ Your site should now show as secure: https://$DOMAIN"
  fi
else
  echo ""
  echo "### Error: Failed to obtain certificate"
  echo "### Check the logs above for details"
  exit 1
fi

