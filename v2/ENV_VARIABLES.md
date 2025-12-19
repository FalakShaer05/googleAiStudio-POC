# Environment Variables

This document lists all required and optional environment variables for the Character Generator application.

## Required Environment Variables

### Google Gemini API
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```
Required for character generation using Google Gemini API.

### AWS S3 Configuration
```bash
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
S3_BUCKET=mosida
```
Required for uploading images to S3. The bucket name should match your S3 bucket.

### CloudFront CDN URL
```bash
CLOUDFRONT_URL=https://d2s4ngnid78ki4.cloudfront.net
```
Required for serving images via CloudFront CDN. This is the CloudFront distribution URL.

### Freepik API (for background removal)
```bash
FREEPIK_API_KEY=your_freepik_api_key_here
```
Required if you want to use the `/remove-bg` endpoint for background removal.

## Optional Environment Variables

### AWS Region
```bash
AWS_DEFAULT_REGION=us-east-1
```
AWS region for S3 operations. Defaults to `us-east-1` if not specified.

### S3 Prefix
```bash
S3_PREFIX=converted/
```
Optional prefix/folder path in S3 bucket. Defaults to `converted/` if not specified.

### Local Path Matching (Fallback)
```bash
LIGHTX_IMAGE_BASE_PATH=outputs
```
Optional local path for fallback image serving if S3 is not configured.

## Example .env File

Create a `.env` file in the project root with the following content:

```bash
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# AWS S3 Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=mosida
S3_PREFIX=converted/

# CloudFront CDN URL
CLOUDFRONT_URL=https://d2s4ngnid78ki4.cloudfront.net

# Freepik API
FREEPIK_API_KEY=your_freepik_api_key_here
```

## Notes

- All generated images are automatically uploaded to S3 and served via CloudFront CDN
- The application will return both `image_url` (CloudFront URL) and `local_path` (local fallback) in API responses
- If S3 upload fails, the application will still work but will only return local paths
- Make sure your S3 bucket has proper permissions and CloudFront is configured to serve from it

