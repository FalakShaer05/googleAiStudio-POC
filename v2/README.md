# Character Generator v2

A Flask-based character generation API using Google Gemini AI and Freepik background removal.

## Features

- Character generation from selfies using Google Gemini AI
- Background removal using Freepik API
- Automatic S3 upload and CloudFront CDN serving
- Support for file uploads and URLs
- Docker support

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
   # Copy from root .env or create new one
   cp ../.env .env  # If .env is in root
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

## API Endpoints

### Character Generation
- `POST /generate-character-web` - Generate character from selfie
- `POST /api/generate-character` - API alias

### Background Removal
- `POST /remove-bg` - Remove background from image

### Health Check
- `GET /health` - Health check endpoint

## Environment Variables

See [ENV_VARIABLES.md](./ENV_VARIABLES.md) for complete list of required environment variables.

## API Testing

See [API_TESTING.md](./API_TESTING.md) for cURL examples and testing instructions.

## Project Structure

```
v2/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── utils/
│   ├── character_utils.py  # Character generation utilities
│   ├── bg_remover.py       # Background removal utilities
│   └── s3_utils.py         # S3 and CloudFront utilities
├── templates/
│   └── index.html         # Web UI
├── uploads/              # Temporary upload directory
└── outputs/              # Generated images directory
```

## Docker Commands

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

## Notes

- The `.env` file should be in the root directory (not in v2/)
- Docker will mount the root `.env` file to `/app/.env` in the container
- Uploads and outputs are persisted in `v2/uploads/` and `v2/outputs/` directories
- All generated images are automatically uploaded to S3 and served via CloudFront CDN

