#!/bin/bash

# Script to initialize Let's Encrypt certificates for the first time
# Run this script once to obtain the initial SSL certificate

if ! docker compose version > /dev/null 2>&1; then
  echo 'Error: docker compose is not installed or not available.' >&2
  exit 1
fi

DOMAIN="dev-google-ai.mosida.com"
EMAIL="falak@livao.com"  # Change this to your email address
STAGING=1  # Set to 1 if you're testing your setup to avoid hitting request limits

if [ -d "$HOME/letsencrypt/conf/live/$DOMAIN" ]; then
  read -p "Existing data found for $DOMAIN. Continue and replace the certificate? (y/N) " decision
  if [ "$decision" != "Y" ] && [ "$decision" != "y" ]; then
    exit
  fi
fi

if [ ! -e "./docker-compose.yml" ]; then
  echo "Error: docker-compose.yml not found in current directory"
  exit 1
fi

echo "### Starting nginx (HTTP only, for Let's Encrypt challenge) ..."
# Start nginx with HTTP only - we'll add HTTPS after getting certificates
docker compose --profile production up -d nginx
echo

# Wait for nginx to be ready
echo "### Waiting for nginx to be ready ..."
sleep 5

# Check if nginx is running
if ! docker compose ps nginx | grep -q "Up"; then
  echo "Error: nginx failed to start. Check logs with: docker compose logs nginx"
  exit 1
fi

# Verify nginx is listening on port 80
echo "### Verifying nginx is accessible on port 80 ..."
if ! curl -f -s http://localhost/.well-known/acme-challenge/test > /dev/null 2>&1; then
  echo "Warning: nginx may not be accessible on port 80"
  echo "Please ensure:"
  echo "  1. Port 80 is open in your firewall"
  echo "  2. No other service is using port 80"
  echo "  3. Check nginx logs: docker compose logs nginx"
fi
echo

echo "### Requesting Let's Encrypt certificate for $DOMAIN ..."
# Select appropriate email arg
case "$EMAIL" in
  "") email_arg="--register-unsafely-without-email" ;;
  *) email_arg="--email $EMAIL" ;;
esac

# Enable staging mode if needed
if [ $STAGING != "0" ]; then staging_arg="--staging"; fi

docker compose run --rm --entrypoint "\
  certbot certonly --webroot -w /var/www/certbot \
    $staging_arg \
    $email_arg \
    -d $DOMAIN \
    --rsa-key-size 4096 \
    --agree-tos \
    --force-renewal" certbot
echo

echo "### Reloading nginx to enable HTTPS ..."
# Test nginx config first
docker compose exec nginx nginx -t
if [ $? -eq 0 ]; then
  docker compose exec nginx nginx -s reload
  echo "### Certificate obtained successfully!"
  echo "### Your site should now be available at https://$DOMAIN"
else
  echo "### Warning: nginx configuration test failed. HTTPS may not work."
  echo "### Check nginx logs: docker compose logs nginx"
fi
echo ""
echo "### Note: If you used --staging, you'll need to run this script again with STAGING=0"
echo "### to get a production certificate."

