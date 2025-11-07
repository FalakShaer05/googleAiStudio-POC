#!/bin/bash

# Script to check certificate details

DOMAIN="dev-google-ai.mosida.com"

echo "### Checking certificate details ..."
echo ""

# Check certificate issuer
echo "### Certificate Issuer:"
docker compose exec nginx openssl x509 -in /etc/letsencrypt/live/$DOMAIN/cert.pem -noout -issuer 2>/dev/null

echo ""
echo "### Certificate Subject:"
docker compose exec nginx openssl x509 -in /etc/letsencrypt/live/$DOMAIN/cert.pem -noout -subject 2>/dev/null

echo ""
echo "### Certificate Validity:"
docker compose exec nginx openssl x509 -in /etc/letsencrypt/live/$DOMAIN/cert.pem -noout -dates 2>/dev/null

echo ""
echo "### Certificate Chain:"
docker compose exec nginx openssl x509 -in /etc/letsencrypt/live/$DOMAIN/chain.pem -noout -issuer 2>/dev/null

echo ""
echo "### Checking if this is a staging certificate:"
ISSUER=$(docker compose exec nginx openssl x509 -in /etc/letsencrypt/live/$DOMAIN/cert.pem -noout -issuer 2>/dev/null)
if echo "$ISSUER" | grep -q "Fake LE Intermediate"; then
    echo "⚠️  WARNING: This is a STAGING certificate (not trusted by browsers)"
    echo "   You need to get a production certificate by running:"
    echo "   1. Edit init-letsencrypt.sh and set STAGING=0"
    echo "   2. Run ./init-letsencrypt.sh again"
else
    echo "✓ This appears to be a production certificate"
fi

echo ""
echo "### Full certificate chain check:"
docker compose exec nginx cat /etc/letsencrypt/live/$DOMAIN/fullchain.pem | openssl x509 -noout -text 2>/dev/null | grep -A 5 "Issuer:" | head -5

