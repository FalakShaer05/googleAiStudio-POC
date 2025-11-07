# SSL Setup Guide for dev-google-ai.mosida.com

This guide explains how to set up free SSL certificates using Let's Encrypt with Docker.

## Prerequisites

✅ Domain `dev-google-ai.mosida.com` is pointing to your server IP  
✅ Ports 80 and 443 are open and accessible from the internet  
✅ Docker and Docker Compose are installed  

## Quick Start

### Step 1: Update Email Address

Before running the initialization script, update the email address in `init-letsencrypt.sh`:

```bash
EMAIL="admin@mosida.com"  # Change this to your email address
```

### Step 2: Initial Certificate Setup

Run the initialization script to obtain your first SSL certificate:

```bash
./init-letsencrypt.sh
```

**Note:** The script uses staging mode by default (`STAGING=1`) to avoid hitting Let's Encrypt rate limits during testing. After successful testing, set `STAGING=0` in the script and run it again to get a production certificate.

### Step 3: Start Services with Production Profile

Start all services including nginx and certbot:

```bash
docker-compose --profile production up -d
```

This will:
- Start your Flask application
- Start nginx reverse proxy (listening on ports 80 and 443)
- Start certbot container for automatic certificate renewal

### Step 4: Verify SSL

Visit your site:
- **HTTPS**: https://dev-google-ai.mosida.com
- **HTTP**: http://dev-google-ai.mosida.com (will redirect to HTTPS)

## How It Works

### Architecture

1. **Nginx**: Reverse proxy that handles:
   - HTTP (port 80) - redirects to HTTPS and serves Let's Encrypt challenges
   - HTTPS (port 443) - serves your application with SSL

2. **Certbot**: Automatically renews certificates every 12 hours (checks if renewal is needed)

3. **Shared Volumes**: Certificates are stored in Docker volumes shared between nginx and certbot

### Automatic Renewal

The certbot container runs continuously and:
- Checks for certificate renewal every 12 hours
- Automatically renews certificates when they're within 30 days of expiration
- Nginx automatically reloads when certificates are renewed

### Manual Renewal

If you need to manually renew certificates:

```bash
./renew-cert.sh
```

## Configuration Details

### Docker Compose Services

- **image-converter**: Your Flask application (port 5000 internally)
- **nginx**: Reverse proxy with SSL (ports 80, 443 externally)
- **certbot**: Certificate management and auto-renewal

### Nginx Configuration

- **HTTP Server**: Handles Let's Encrypt challenges and redirects to HTTPS
- **HTTPS Server**: Serves your application with modern SSL/TLS configuration
- **Security Headers**: HSTS, X-Frame-Options, X-Content-Type-Options, etc.

### SSL Configuration

- **Protocols**: TLSv1.2 and TLSv1.3 only
- **Ciphers**: Modern, secure cipher suites
- **Certificate**: Let's Encrypt (valid for 90 days, auto-renewed)

## Troubleshooting

### Certificate Not Obtained

1. **Check DNS**: Ensure `dev-google-ai.mosida.com` resolves to your server IP
   ```bash
   nslookup dev-google-ai.mosida.com
   ```

2. **Check Ports**: Ensure ports 80 and 443 are open
   ```bash
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   ```

3. **Check Logs**: View certbot logs
   ```bash
   docker-compose logs certbot
   ```

### Nginx Not Starting

1. **Check Configuration**: Test nginx config
   ```bash
   docker-compose exec nginx nginx -t
   ```

2. **Check Logs**: View nginx logs
   ```bash
   docker-compose logs nginx
   ```

### Certificate Renewal Issues

1. **Check Certbot Status**: View certbot container logs
   ```bash
   docker-compose logs certbot
   ```

2. **Manual Renewal Test**: Test renewal manually
   ```bash
   docker-compose run --rm certbot renew --dry-run
   ```

## Important Notes

1. **First Time Setup**: Use staging mode first (`STAGING=1`) to test, then switch to production (`STAGING=0`)

2. **Email Address**: Update the email in `init-letsencrypt.sh` - Let's Encrypt will send renewal reminders

3. **Port 5000**: You can now remove the direct port 5000 exposure from docker-compose.yml if you want, as nginx handles all external traffic

4. **Production Profile**: Always use `--profile production` when starting services to include nginx and certbot

## Commands Reference

```bash
# Start all services (with SSL)
docker-compose --profile production up -d

# Stop all services
docker-compose --profile production down

# View logs
docker-compose --profile production logs -f

# Restart nginx
docker-compose --profile production restart nginx

# Check certificate expiration
docker-compose run --rm certbot certificates

# Test certificate renewal (dry run)
docker-compose run --rm certbot renew --dry-run
```

## Security Best Practices

✅ SSL/TLS 1.2 and 1.3 only  
✅ Modern cipher suites  
✅ HSTS header enabled  
✅ Security headers configured  
✅ Automatic certificate renewal  
✅ HTTP to HTTPS redirect  

Your site is now secured with free SSL from Let's Encrypt! 🔒

