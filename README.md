# Image Processing API

A RESTful API for image-to-image conversion using Google Gemini AI, with optional background merging capabilities. This service provides both a web interface and a programmatic API for caricature generation and background compositing.

## Features

- **Caricature Generation**: Transform images using AI-powered text prompts
- **Background Merging**: Professional background removal and compositing
- **Web Interface**: User-friendly web application
- **REST API**: Programmatic access for integrations
- **Secure Authentication**: API key-based access control
- **Rate Limiting**: Configurable request limits per API key
- **Auto Cleanup**: Automatic file cleanup after processing
- **Multiple Formats**: Support for PNG, JPG, JPEG, GIF, BMP, TIFF

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp env.example .env

# Edit .env and add your Google API key
# GOOGLE_API_KEY=your-google-api-key-here

# Setup API keys (optional, for API access)
python setup_api.py

# Start server
python start_server.py
```

### 2. Access the Application

- **Web Interface**: http://localhost:5000
- **API Health Check**: http://localhost:5000/api/v1/health

### 3. Test API (Optional)

```bash
# Run test suite
python test_api.py

# Test CORS in browser
open test_cors.html
```

## Web Interface

The web application provides a user-friendly interface for:

- Uploading images for conversion
- Entering transformation prompts
- Adding background images for merging
- Adjusting position, scale, and opacity
- Downloading processed results

## API Usage

### CORS Support

The API supports Cross-Origin Resource Sharing (CORS) for web applications:

- **Allowed Origins**: All origins (`*`)
- **Allowed Methods**: GET, POST, PUT, DELETE, OPTIONS
- **Allowed Headers**: Content-Type, Authorization, X-API-Key
- **Credentials**: Supported

### Authentication

All API endpoints require authentication via API key. Provide the API key in the header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:5000/api/v1/health
```

### Main Endpoint: Process Image

```http
POST /api/v1/process
```

**Parameters:**
- `image` (file, required*): Input image file
- `image_url` (string, required*): URL of input image to download
- `prompt` (string, required): Transformation prompt
- `background` (file, optional): Background image file
- `background_url` (string, optional): URL of background image to download
- `position` (string, optional): Position for merging (default: 'center')
- `scale` (float, optional): Scale factor 0.1-3.0 (default: 1.0)
- `opacity` (float, optional): Opacity 0.0-1.0 (default: 1.0)
- `canvas_size` (string, optional): Print size (e.g., '8x10')

*Either `image` or `image_url` is required, but not both.

**Example Request (File Upload):**

```bash
curl -X POST "http://localhost:5000/api/v1/process" \
  -H "X-API-Key: your-api-key" \
  -F "image=@input.jpg" \
  -F "prompt=Transform this into a cartoon caricature" \
  -F "background=@background.jpg" \
  -F "position=center" \
  -F "scale=1.2"
```

**Example Request (Image URL):**

```bash
curl -X POST "http://localhost:5000/api/v1/process" \
  -H "X-API-Key: your-api-key" \
  -F "image_url=https://example.com/image.jpg" \
  -F "prompt=Transform this into a cartoon caricature" \
  -F "background_url=https://example.com/background.jpg" \
  -F "position=center" \
  -F "scale=1.2"
```

**Success Response:**
```json
{
  "success": true,
  "message": "Image processed successfully",
  "output_url": "http://localhost:5000/api/v1/download/processed_image.jpg",
  "processing_time": "12.5s",
  "file_size": "2.1MB",
  "metadata": {
    "image_info": {
      "width": 1920,
      "height": 1080,
      "mode": "RGB",
      "format": "JPEG"
    },
    "prompt_used": "Transform this into a cartoon caricature",
    "background_merged": true,
    "position": "center",
    "scale": 1.0,
    "opacity": 1.0
  }
}
```

### Other API Endpoints

- `GET /api/v1/health` - Health check
- `POST /api/v1/validate` - Validate API key
- `GET /api/v1/download/{filename}` - Download processed image
- `GET /api/v1/status` - Usage statistics

## Programming Examples

### Python

```python
import requests

def process_image(image_path, prompt, background_path=None, api_key="your-api-key"):
    url = "http://localhost:5000/api/v1/process"
    headers = {"X-API-Key": api_key}
    
    files = {"image": open(image_path, "rb")}
    data = {"prompt": prompt}
    
    if background_path:
        files["background"] = open(background_path, "rb")
        data.update({
            "position": "center",
            "scale": "1.0",
            "opacity": "1.0"
        })
    
    response = requests.post(url, headers=headers, files=files, data=data)
    return response.json()

# Usage
result = process_image("input.jpg", "Transform into cartoon caricature")
if result["success"]:
    print(f"Success! Download: {result['output_url']}")
```

### JavaScript

```javascript
async function processImage(imageFile, prompt, backgroundFile = null, apiKey = "your-api-key") {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('prompt', prompt);
    
    if (backgroundFile) {
        formData.append('background', backgroundFile);
        formData.append('position', 'center');
        formData.append('scale', '1.0');
        formData.append('opacity', '1.0');
    }
    
    const response = await fetch('http://localhost:5000/api/v1/process', {
        method: 'POST',
        headers: {'X-API-Key': apiKey},
        body: formData
    });
    
    return await response.json();
}

// Usage
const fileInput = document.getElementById('imageInput');
const result = await processImage(fileInput.files[0], 'Transform into cartoon caricature');
console.log(result);
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Required
GOOGLE_API_KEY=your-google-api-key

# API Configuration (optional)
API_KEYS=dev-key-123,prod-key-456
API_RATE_LIMIT_REQUESTS=100
API_RATE_LIMIT_WINDOW_MINUTES=60

# Optional
API_FILE_CLEANUP_HOURS=24
API_MAX_FILE_SIZE_MB=16
```

### Rate Limiting

- Default: 100 requests per hour per API key
- Configurable via `API_RATE_LIMIT_REQUESTS`
- Window configurable via `API_RATE_LIMIT_WINDOW_MINUTES`

## Error Handling

The API returns standardized error responses:

```json
{
  "success": false,
  "error": "Invalid API key",
  "error_code": "AUTH_002",
  "details": "API key not found or invalid",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Common Error Codes

- `AUTH_001`: API key required
- `AUTH_002`: Invalid API key
- `RATE_001`: Rate limit exceeded
- `VALIDATION_001`: Missing required parameter
- `VALIDATION_002`: Invalid file type
- `VALIDATION_003`: File too large
- `PROCESSING_001`: Image conversion failed
- `PROCESSING_002`: Background removal failed
- `PROCESSING_003`: Image compositing failed

## File Management

- **Upload Limit**: 16MB maximum file size
- **Supported Formats**: PNG, JPG, JPEG, GIF, BMP, TIFF
- **Auto Cleanup**: Files deleted after 24 hours
- **Temporary Storage**: Processed files stored in `outputs/` directory

## Security

- **API Key Authentication**: Secure token-based access
- **Input Validation**: File type and size validation
- **Rate Limiting**: Prevents abuse
- **File Sanitization**: Safe filename handling
- **Error Logging**: Comprehensive error tracking

## Docker Deployment

### Quick Start with Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the application
# Web Interface: http://localhost:80
# API: http://localhost:80/api/v1/health
```

### Docker Commands

```bash
# Build the image
docker build -t image-converter .

# Run the container
docker run -p 5000:5000 --env-file .env image-converter

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

## Production Deployment

### Environment Setup

1. Set production API keys
2. Configure rate limits
3. Set up monitoring
4. Enable HTTPS
5. Configure backup storage

### Monitoring

- Health check endpoint: `/api/v1/health`
- Usage statistics: `/api/v1/status`
- Error logging: Check application logs
- File cleanup: Automatic after 24 hours

## Troubleshooting

### Common Issues

1. **"API key required"** - Add `X-API-Key` header
2. **"Invalid API key"** - Check your API key in `.env` file
3. **"Rate limit exceeded"** - Wait and retry with exponential backoff
4. **"File too large"** - Reduce image size (max 16MB)
5. **"Invalid file type"** - Use supported formats (PNG, JPG, etc.)

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export FLASK_DEBUG=1
python start_server.py
```

## Development

### Project Structure

```
/
├── app.py                    # Main Flask application
├── api/                      # API module
│   ├── __init__.py          # Module initialization
│   ├── auth.py              # Authentication & rate limiting
│   ├── models.py            # Data models & error codes
│   ├── utils.py             # Utility functions
│   └── endpoints.py         # API endpoints
├── templates/               # Web interface templates
├── static/                  # Static assets
├── uploads/                 # Temporary upload storage
├── outputs/                 # Processed image storage
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
└── README.md               # This file
```

### Adding New Features

1. **API Endpoints**: Add to `api/endpoints.py`
2. **Authentication**: Modify `api/auth.py`
3. **Data Models**: Update `api/models.py`
4. **Utilities**: Add to `api/utils.py`

## License

This project follows the same license as the Google AI Studio examples (Apache 2.0 License).

## Support

For issues and questions:
1. Check the logs for error details
2. Verify API key configuration
3. Test with the provided test script
4. Check network connectivity
5. Review the troubleshooting section above