#!/bin/bash

# Script to manually renew Let's Encrypt certificates
# This is also handled automatically by the certbot container,
# but you can run this manually if needed

echo "### Renewing Let's Encrypt certificates ..."

docker-compose run --rm --entrypoint "\
  certbot renew" certbot

echo "### Reloading nginx ..."
docker-compose exec nginx nginx -s reload

echo "### Certificate renewal complete!"

