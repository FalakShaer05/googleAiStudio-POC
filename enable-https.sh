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
echo "### Checking for SSL certificates ..."
if ! docker compose exec nginx test -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" 2>/dev/null; then
  echo "Error: SSL certificates not found. Run ./init-letsencrypt.sh first."
  exit 1
fi
echo "### Certificates found ✓"

# Create backup
cp "$NGINX_CONF" "${NGINX_CONF}.backup.$(date +%Y%m%d_%H%M%S)"

echo "### Enabling HTTPS in nginx configuration ..."

# Use a temporary file for the output
TMPFILE=$(mktemp)

# Process the file line by line
in_https_block=false
in_http_location=false
http_location_done=false

while IFS= read -r line; do
    # Detect start of HTTPS server block
    if [[ "$line" =~ ^[[:space:]]*#\ server\ \{ ]] && [[ "$(grep -A 5 "$line" "$NGINX_CONF")" =~ "listen 443" ]]; then
        in_https_block=true
        echo "$line" | sed 's/^[[:space:]]*# server {/    server {/' >> "$TMPFILE"
        continue
    fi
    
    # Detect end of HTTPS server block
    if [[ "$in_https_block" == true ]] && [[ "$line" =~ ^[[:space:]]*#\ \} ]]; then
        in_https_block=false
        echo "$line" | sed 's/^[[:space:]]*# }$/    }/' >> "$TMPFILE"
        continue
    fi
    
    # If we're in the HTTPS block, uncomment lines
    if [[ "$in_https_block" == true ]]; then
        # Remove leading comment and adjust indentation
        if [[ "$line" =~ ^[[:space:]]*#\  ]]; then
            # Replace "    # " with "        " (8 spaces)
            echo "$line" | sed 's/^[[:space:]]*# /        /' >> "$TMPFILE"
        elif [[ "$line" =~ ^[[:space:]]*# ]]; then
            echo "$line" | sed 's/^[[:space:]]*#/        /' >> "$TMPFILE"
        else
            echo "$line" >> "$TMPFILE"
        fi
        continue
    fi
    
    # Handle HTTP location / block - change to redirect
    if [[ "$line" =~ "Temporarily serve app over HTTP" ]] && [[ "$http_location_done" == false ]]; then
        echo "$line" >> "$TMPFILE"
        in_http_location=true
        continue
    fi
    
    if [[ "$in_http_location" == true ]] && [[ "$line" =~ ^[[:space:]]*location\ /\ \{ ]]; then
        echo "        # Redirect all HTTP traffic to HTTPS" >> "$TMPFILE"
        echo "        location / {" >> "$TMPFILE"
        echo "            return 301 https://\$host\$request_uri;" >> "$TMPFILE"
        echo "        }" >> "$TMPFILE"
        in_http_location=false
        http_location_done=true
        # Skip the proxy_pass block until we find the closing brace
        skip_proxy_block=true
        continue
    fi
    
    if [[ "$skip_proxy_block" == true ]]; then
        if [[ "$line" =~ ^[[:space:]]*\}$ ]]; then
            skip_proxy_block=false
            # Don't echo this brace, we already added it
        fi
        continue
    fi
    
    # Default: just echo the line
    echo "$line" >> "$TMPFILE"
    
done < "$NGINX_CONF"

# Replace the original file
mv "$TMPFILE" "$NGINX_CONF"

echo "### Configuration updated"

echo "### Testing nginx configuration ..."
docker compose exec nginx nginx -t

if [ $? -eq 0 ]; then
  echo "### Reloading nginx ..."
  docker compose exec nginx nginx -s reload
  echo ""
  echo "### HTTPS enabled successfully! ✓"
  echo "### Your site is now available at https://$DOMAIN"
  echo "### HTTP will automatically redirect to HTTPS"
else
  echo "### Error: nginx configuration test failed"
  echo "### Restoring backup..."
  LATEST_BACKUP=$(ls -t ${NGINX_CONF}.backup.* 2>/dev/null | head -1)
  if [ -n "$LATEST_BACKUP" ]; then
    mv "$LATEST_BACKUP" "$NGINX_CONF"
  fi
  exit 1
fi
