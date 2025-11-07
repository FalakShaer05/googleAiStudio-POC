# SSL Setup Troubleshooting Guide

## Issue: Connection Refused Error

If you see this error from Let's Encrypt:
```
Connection refused
Detail: Fetching http://dev-google-ai.mosida.com/.well-known/acme-challenge/...
```

This means Let's Encrypt cannot reach your server on port 80. Follow these steps:

### Step 1: Check if Port 80 is Open

```bash
# Check if port 80 is listening
sudo netstat -tlnp | grep :80
# or
sudo ss -tlnp | grep :80

# Check if nginx is running
docker compose ps nginx
```

### Step 2: Check Firewall

```bash
# If using UFW (Ubuntu)
sudo ufw status
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# If using firewalld (CentOS/RHEL)
sudo firewall-cmd --list-ports
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload

# If using iptables
sudo iptables -L -n | grep 80
```

### Step 3: Test from Outside

Test if port 80 is accessible from the internet:

```bash
# From another machine or use an online tool
curl -I http://dev-google-ai.mosida.com

# Should return HTTP headers, not connection refused
```

### Step 4: Check Nginx Logs

```bash
# Check nginx error logs
docker compose logs nginx

# Check if nginx is serving the challenge path
curl http://localhost/.well-known/acme-challenge/test
```

### Step 5: Verify DNS

Ensure your domain points to the correct IP:

```bash
# Check DNS resolution
nslookup dev-google-ai.mosida.com
dig dev-google-ai.mosida.com

# Should return your server's public IP
```

### Step 6: Check for Port Conflicts

```bash
# Check what's using port 80
sudo lsof -i :80
# or
sudo fuser 80/tcp

# If something else is using port 80, stop it or change nginx port mapping
```

## Issue: Nginx Container Restarting

If nginx keeps restarting:

1. **Check logs:**
   ```bash
   docker compose logs nginx
   ```

2. **Test nginx configuration:**
   ```bash
   docker compose exec nginx nginx -t
   ```

3. **Common causes:**
   - SSL certificates don't exist (HTTPS block is active)
   - Configuration syntax error
   - Port already in use

## Issue: Certificate Obtained but HTTPS Not Working

After successfully obtaining certificates:

1. **Uncomment HTTPS block in nginx.conf:**
   - Remove `#` from all lines in the HTTPS server block (lines 108-170)
   - Change HTTP location `/` to redirect to HTTPS

2. **Or use the enable script:**
   ```bash
   ./enable-https.sh
   ```

3. **Reload nginx:**
   ```bash
   docker compose exec nginx nginx -s reload
   ```

## Quick Diagnostic Commands

```bash
# Check all services status
docker compose --profile production ps

# Check nginx status
docker compose logs nginx --tail 50

# Check certbot status
docker compose logs certbot --tail 50

# Test nginx config
docker compose exec nginx nginx -t

# Check if certificates exist
docker compose exec nginx ls -la /etc/letsencrypt/live/dev-google-ai.mosida.com/

# Test HTTP access
curl -I http://dev-google-ai.mosida.com

# Test HTTPS access (after enabling)
curl -I https://dev-google-ai.mosida.com
```

## Common Solutions

### Solution 1: Port 80 Blocked by Firewall

```bash
# Ubuntu/Debian
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw reload

# CentOS/RHEL
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

### Solution 2: Another Service Using Port 80

```bash
# Find what's using port 80
sudo lsof -i :80

# Stop Apache if running
sudo systemctl stop apache2  # Ubuntu/Debian
sudo systemctl stop httpd     # CentOS/RHEL

# Or change the conflicting service's port
```

### Solution 3: Nginx Not Starting

```bash
# Check nginx logs
docker compose logs nginx

# Restart nginx
docker compose restart nginx

# If still failing, check configuration
docker compose exec nginx nginx -t
```

### Solution 4: DNS Not Propagated

Wait a few minutes for DNS to propagate, then verify:

```bash
nslookup dev-google-ai.mosida.com
# Should return your server IP
```

## Still Having Issues?

1. Check all logs: `docker compose --profile production logs`
2. Verify domain DNS: `nslookup dev-google-ai.mosida.com`
3. Test port accessibility from outside your server
4. Ensure no other web server is running on port 80
5. Check server firewall rules

