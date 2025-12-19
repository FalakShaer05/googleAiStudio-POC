# Character Generator API

A Flask-based character generation API using Google Gemini AI and Freepik background removal. This service provides both a web interface and a programmatic API for character generation, background removal, and image compositing with automatic S3 upload and CloudFront CDN serving.

## Features

- **Character Generation**: Transform selfies into characters using Google Gemini AI with identity preservation
- **Background Removal**: Remove backgrounds from images using Freepik API
- **Automatic S3 Upload**: All generated images are automatically uploaded to S3
- **CloudFront CDN**: Images are served via CloudFront CDN for fast global delivery
- **Flexible Input**: Support for both file uploads and image URLs
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Web Interface**: User-friendly web application for testing
- **REST API**: Programmatic access for integrations

## Quick Start

### Using Docker (Recommended)

1. **Set up environment variables** in the root `.env` file:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key
   AWS_ACCESS_KEY_ID=your_aws_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret
   S3_BUCKET=mosida
   CLOUDFRONT_URL=https://d2s4ngnid78ki4.cloudfront.net
   FREEPIK_API_KEY=your_freepik_key
   ```

2. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

3. **Access the application**:
   - Web UI: http://localhost:5000
   - API: http://localhost:5000/generate-character-web
   - Health: http://localhost:5000/health

### Local Development

1. **Create virtual environment**:
   ```bash
   cd v2
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (create `.env` in root directory):
   ```bash
   # The .env file should be in the root directory
   # See Environment Variables section below for required variables
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

## Environment Variables

### Required Environment Variables

#### Google Gemini API
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```
Required for character generation using Google Gemini API.

#### AWS S3 Configuration
```bash
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
S3_BUCKET=mosida
```
Required for uploading images to S3. The bucket name should match your S3 bucket.

#### CloudFront CDN URL
```bash
CLOUDFRONT_URL=https://d2s4ngnid78ki4.cloudfront.net
```
Required for serving images via CloudFront CDN. This is the CloudFront distribution URL.

#### Freepik API (for background removal)
```bash
FREEPIK_API_KEY=your_freepik_api_key_here
```
Required if you want to use the `/remove-bg` endpoint for background removal.

### Optional Environment Variables

#### AWS Region
```bash
AWS_DEFAULT_REGION=us-east-1
```
AWS region for S3 operations. Defaults to `us-east-1` if not specified.

#### S3 Prefix
```bash
S3_PREFIX=converted/
```
Optional prefix/folder path in S3 bucket. Defaults to `converted/` if not specified.

#### Local Path Matching (Fallback)
```bash
LIGHTX_IMAGE_BASE_PATH=outputs
```
Optional local path for fallback image serving if S3 is not configured.

### Example .env File

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

## API Endpoints

### Character Generation

#### Generate Character
```http
POST /generate-character-web
POST /api/generate-character  # API alias
```

**Parameters:**
- `selfie` (file) OR `selfie_url` (string, required): Selfie/reference photo
- `character_prompt` (string, required): Description of character transformation
- `background` (file, optional) OR `background_url` (string, optional): Background image
- `position` (string, optional): Position for compositing - 'center', 'bottom' (default: 'bottom')
- `scale` (float, optional): Scale factor 0.1-3.0 (default: 1.0)
- `canvas_size` (string, optional): Print size (e.g., '8x10', '11x14', '16x20')
- `dpi` (int, optional): DPI for print (default: 300)

**Success Response:**
```json
{
  "success": true,
  "message": "Character generated successfully",
  "output_filename": "output_abc123_composited_selfie_xyz.png",
  "image_url": "https://d2s4ngnid78ki4.cloudfront.net/converted/uuid_filename.png",
  "local_path": "/outputs/output_abc123_composited_selfie_xyz.png",
  "metadata": {
    "image_info": {
      "width": 1024,
      "height": 1024,
      "mode": "RGB",
      "format": "PNG"
    },
    "character_prompt": "...",
    "position": "bottom",
    "scale": 1.0
  }
}
```

### Background Removal

#### Remove Background
```http
POST /remove-bg
```

**Parameters:**
- `image` (file) OR `image_url` (string, required): Image to process

**Success Response:**
```json
{
  "success": true,
  "message": "Background removed successfully",
  "output_filename": "output_abc123_bg_removed_image_xyz.png",
  "image_url": "https://d2s4ngnid78ki4.cloudfront.net/converted/uuid_filename.png",
  "local_path": "/outputs/output_abc123_bg_removed_image_xyz.png",
  "metadata": {
    "image_info": {
      "width": 1024,
      "height": 1024,
      "mode": "RGBA",
      "format": "PNG"
    }
  }
}
```

### Health Check

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "ok"
}
```

## API Testing with cURL

### Base URL
```
http://localhost:5000
```

### 1. Health Check

```bash
curl -X GET http://localhost:5000/health
```

### 2. Character Generation

#### With Selfie File and Background File
```bash
curl -X POST http://localhost:5000/generate-character-web \
  -F "selfie=@/path/to/your/selfie.jpg" \
  -F "background=@/path/to/your/background.jpg" \
  -F "character_prompt=A full-body cartoon caricature with bright colors" \
  -F "position=bottom" \
  -F "scale=1.0"
```

#### With Selfie URL and Background URL
```bash
curl -X POST http://localhost:5000/generate-character-web \
  -F "selfie_url=https://example.com/selfie.jpg" \
  -F "background_url=https://example.com/background.jpg" \
  -F "character_prompt=A cartoon character with vibrant colors" \
  -F "position=bottom" \
  -F "scale=1.0"
```

#### With Selfie Only (No Background)
```bash
curl -X POST http://localhost:5000/generate-character-web \
  -F "selfie=@/path/to/your/selfie.jpg" \
  -F "character_prompt=A heroic fantasy character in detailed digital art style" \
  -F "position=center" \
  -F "scale=1.0"
```

### 3. Background Removal

#### With Image File Upload
```bash
curl -X POST http://localhost:5000/remove-bg \
  -F "image=@/path/to/your/image.jpg"
```

#### With Image URL
```bash
curl -X POST http://localhost:5000/remove-bg \
  -F "image_url=https://example.com/image.jpg"
```

### 4. Download Generated Image

```bash
curl -X GET http://localhost:5000/download/output_filename.png \
  --output downloaded_image.png
```

### Error Responses

#### Missing Required Field
```json
{
  "error": "Selfie image is required"
}
```

#### Invalid File Type
```json
{
  "error": "Invalid selfie file type"
}
```

#### Generation Failed
```json
{
  "error": "Character generation failed: No image generated by Gemini"
}
```

## Docker Deployment

### Docker Commands

```bash
# Build image
docker-compose build

# Start services
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up --build --force-recreate
```

### Docker Configuration

- **Service Name**: `character-generator`
- **Container Name**: `character-generator-app`
- **Port**: `5000:5000`
- **Volumes**:
  - `./v2/uploads:/app/uploads` - Upload directory
  - `./v2/outputs:/app/outputs` - Output directory
  - `./.env:/app/.env:ro` - Environment variables (read-only)

### Production Deployment with Nginx

The `docker-compose.yml` includes optional Nginx reverse proxy and Certbot for SSL certificates:

```bash
# Start with production profile (includes Nginx and SSL)
docker-compose --profile production up -d
```

## Project Structure

```
googleAiStudio-POC/
├── .env                      # Environment variables (root directory)
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile                # Docker image definition
├── Dockerfile.fast           # Fast Docker build (minimal dependencies)
├── README.md                 # This file
└── v2/                       # Main application directory
    ├── app.py                # Main Flask application
    ├── requirements.txt      # Python dependencies
    ├── utils/
    │   ├── character_utils.py  # Character generation utilities
    │   ├── bg_remover.py       # Background removal utilities
    │   └── s3_utils.py         # S3 and CloudFront utilities
    ├── templates/
    │   └── index.html        # Web UI
    ├── uploads/              # Temporary upload directory
    └── outputs/              # Generated images directory
```

## Notes

- The `.env` file should be in the root directory (not in v2/)
- Docker will mount the root `.env` file to `/app/.env` in the container
- Uploads and outputs are persisted in `v2/uploads/` and `v2/outputs/` directories
- All generated images are automatically uploaded to S3 and served via CloudFront CDN
- The application will return both `image_url` (CloudFront URL) and `local_path` (local fallback) in API responses
- If S3 upload fails, the application will still work but will only return local paths
- Make sure your S3 bucket has proper permissions and CloudFront is configured to serve from it

## Troubleshooting

### Common Issues

1. **"FREEPIK_API_KEY not found"** - Add `FREEPIK_API_KEY` to your `.env` file
2. **"S3 upload skipped: missing env vars"** - Check AWS credentials in `.env`
3. **"Background removal failed"** - Verify Freepik API key is valid
4. **"Character generation failed"** - Check Gemini API key and image accessibility
5. **Docker build fails** - Ensure you're running from the root directory

### Debug Mode

Enable debug logging:
```bash
export FLASK_DEBUG=1
python v2/app.py
```

### Check Logs

```bash
# Docker logs
docker-compose logs -f character-generator

# Local logs
# Check console output when running python app.py
```

## Support

For issues and questions:
1. Check the logs for error details
2. Verify environment variables in `.env` file
3. Test with the provided cURL examples
4. Check network connectivity
5. Review the troubleshooting section above

## License

This project follows the same license as the Google AI Studio examples (Apache 2.0 License).
