# Image-to-Image Conversion with Google Gemini API

This project demonstrates how to use Google's Gemini API for image-to-image conversion, allowing you to transform existing images using text prompts.

## Features

- **Image-to-Image Conversion**: Transform existing images using natural language prompts
- **Multiple Input Formats**: Supports various image formats (PNG, JPEG, etc.)
- **Flexible Prompts**: Use descriptive text to guide the transformation
- **Error Handling**: Robust error handling and validation
- **Command Line Interface**: Easy to use from command line

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Google AI API Key**:
   - Get your API key from [Google AI Studio](https://ai.google.dev/)
   - Copy the example environment file:
     ```bash
     cp env.example .env
     ```
   - Edit the `.env` file and add your API key:
     ```
     GOOGLE_API_KEY=your-actual-api-key-here
     ```

## Usage

### Basic Usage

```python
from image_to_image_converter import convert_image_to_image

# Convert an image with a text prompt
success = convert_image_to_image(
    input_image_path="input.jpg",
    prompt="Transform this into a watercolor painting",
    output_path="output.png"
)
```

### Command Line Usage

```bash
# Basic usage
python image_to_image_converter.py /path/to/your/image.jpg

# With custom prompt
python image_to_image_converter.py /path/to/your/image.jpg "Make this image look like a vintage photograph"

# Using the example from Google's documentation
python image_to_image_converter.py cat_image.png "Create a picture of my cat eating a nano-banana in a fancy restaurant under the Gemini constellation"
```

## Example Prompts

Here are some example prompts you can use for different transformations:

- **Artistic Styles**: "Transform this into a Van Gogh painting"
- **Color Changes**: "Make this image black and white with dramatic lighting"
- **Style Transfer**: "Apply a cyberpunk aesthetic to this image"
- **Object Addition**: "Add a beautiful sunset in the background"
- **Scene Changes**: "Transform this into a fantasy forest scene"

## How It Works

The script uses Google's Gemini 2.5 Flash Image Preview model, which can:

1. **Accept both text and image inputs** simultaneously
2. **Understand the context** of your existing image
3. **Apply transformations** based on your text prompt
4. **Generate a new image** that combines your input with the requested changes

## API Reference

Based on the [Google AI Studio documentation](https://ai.google.dev/gemini-api/docs/image-generation#python), this implementation uses:

- **Model**: `gemini-2.5-flash-image-preview`
- **Input**: Text prompt + existing image
- **Output**: Generated image with SynthID watermark

## Limitations

- All generated images include a SynthID watermark
- Best performance with English, Spanish, Japanese, Chinese, and Hindi prompts
- Works best with up to 3 images as input
- Image generation does not support audio or video inputs

## Error Handling

The script includes comprehensive error handling for:
- Missing input images
- Invalid file paths
- API connection issues
- Invalid image formats

## Requirements

- Python 3.7+ (for local development)
- Google AI API key (stored in `.env` file)
- Internet connection for API calls
- Docker & Docker Compose (for containerized deployment)

## Environment Setup

The project uses a `.env` file to securely store your API key:

1. **Copy the example file**:
   ```bash
   cp env.example .env
   ```

2. **Edit `.env` file** with your actual API key:
   ```
   GOOGLE_API_KEY=your-actual-api-key-here
   ```

3. **Never commit the `.env` file** to version control (it's already in `.gitignore`)

## Docker Deployment

### Quick Start with Docker

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd googleAiStudio-POC
   cp env.example .env
   # Edit .env with your API key
   ```

2. **Run with Docker Compose**:
   ```bash
   # Development mode (Flask app only)
   docker-compose up --build
   
   # Production mode (with Nginx reverse proxy)
   docker-compose --profile production up --build
   ```

3. **Access the application**:
   - Development: http://localhost:5000
   - Production: http://localhost:80

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

# Rebuild and restart
docker-compose up --build --force-recreate
```

### Docker Features

- **Multi-stage build** for optimized image size
- **Non-root user** for security
- **Health checks** for container monitoring
- **Volume mounts** for persistent file storage
- **Nginx reverse proxy** for production deployment
- **Rate limiting** and security headers
- **Gzip compression** for better performance

## License

This project follows the same license as the Google AI Studio examples (Apache 2.0 License).
