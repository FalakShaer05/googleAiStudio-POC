#!/bin/bash

# Script to check certificate details using certbot

DOMAIN="dev-google-ai.mosida.com"

echo "### Checking certificate details ..."
echo ""

# Use certbot to check certificate
echo "### Certificate Information:"
docker compose run --rm --entrypoint "\
  certbot certificates" certbot 2>/dev/null | grep -A 20 "$DOMAIN"

echo ""
echo "### Checking certificate issuer from certbot:"
ISSUER_INFO=$(docker compose run --rm --entrypoint "\
  certbot certificates" certbot 2>/dev/null | grep -A 10 "$DOMAIN")

if echo "$ISSUER_INFO" | grep -qi "staging\|fake"; then
    echo "⚠️  WARNING: This is a STAGING certificate (not trusted by browsers)"
    echo ""
    echo "   To get a production certificate, run:"
    echo "   ./get-production-cert.sh"
else
    echo "✓ This appears to be a production certificate"
fi

echo ""
echo "### Certificate files location:"
docker compose exec nginx ls -la /etc/letsencrypt/live/$DOMAIN/ 2>/dev/null || echo "Certificates not found"

