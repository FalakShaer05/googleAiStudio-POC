#!/bin/bash

# Script to enable HTTPS after certificates are obtained
# Run this after ./init-letsencrypt.sh succeeds

DOMAIN="dev-google-ai.mosida.com"
NGINX_CONF="./nginx.conf"

if [ ! -f "$NGINX_CONF" ]; then
  echo "Error: nginx.conf not found"
  exit 1
fi

# Check if certificates exist
if ! docker compose exec nginx test -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" 2>/dev/null; then
  echo "Error: SSL certificates not found. Run ./init-letsencrypt.sh first."
  exit 1
fi

echo "### Enabling HTTPS in nginx configuration ..."

# Uncomment HTTPS server block
sed -i.bak 's/^    # server {$/    server {/g; s/^    #     listen 443/        listen 443/g; s/^    #     server_name/        server_name/g; s/^    #     # SSL/        # SSL/g; s/^    #     ssl_/        ssl_/g; s/^    #     # SSL configuration/        # SSL configuration/g; s/^    #     # Security headers/        # Security headers/g; s/^    #     add_header/        add_header/g; s/^    #     # Client max/        # Client max/g; s/^    #     client_max/        client_max/g; s/^    #     # Main application/        # Main application/g; s/^    #     location \/ {$/        location \/ {/g; s/^    #         limit_req/            limit_req/g; s/^    #         proxy_pass/            proxy_pass/g; s/^    #         proxy_set_header/            proxy_set_header/g; s/^    #         proxy_connect_timeout/            proxy_connect_timeout/g; s/^    #         proxy_send_timeout/            proxy_send_timeout/g; s/^    #         proxy_read_timeout/            proxy_read_timeout/g; s/^    #     }$/        }/g; s/^    #     # File upload/        # File upload/g; s/^    #     # Health check/        # Health check/g; s/^    #     # Static files/        # Static files/g; s/^    # }$/    }/g' "$NGINX_CONF"

# Change HTTP location / to redirect to HTTPS
sed -i.bak2 '/^        # Temporarily serve app over HTTP/,/^        }$/ {
  /^        location \/ {$/,/^        }$/ {
    s/^        location \/ {$/        # Redirect all HTTP traffic to HTTPS\n        location \/ {/g
    /^        limit_req zone=api/,/^        proxy_read_timeout 30s;$/ {
      s/^        /        # /g
    }
    /^        }$/ {
      a\
            return 301 https://$host$request_uri;
    }
  }
}' "$NGINX_CONF"

# Clean up backup files
rm -f "$NGINX_CONF.bak" "$NGINX_CONF.bak2" 2>/dev/null

echo "### Testing nginx configuration ..."
docker compose exec nginx nginx -t

if [ $? -eq 0 ]; then
  echo "### Reloading nginx ..."
  docker compose exec nginx nginx -s reload
  echo "### HTTPS enabled successfully!"
  echo "### Your site is now available at https://$DOMAIN"
else
  echo "### Error: nginx configuration test failed"
  echo "### Please check nginx.conf manually"
  exit 1
fi

