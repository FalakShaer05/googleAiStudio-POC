# API Testing with cURL

This document provides cURL commands to test all API endpoints.

## Base URL
```
http://localhost:5000
```

## 1. Health Check

```bash
curl -X GET http://localhost:5000/health
```

**Expected Response:**
```json
{
  "status": "ok"
}
```

## 2. Character Generation

### With Selfie File and Background File

```bash
curl -X POST http://localhost:5000/generate-character-web \
  -F "selfie=@/path/to/your/selfie.jpg" \
  -F "background=@/path/to/your/background.jpg" \
  -F "character_prompt=A full-body, hand-drawn cartoon-style caricature with slightly exaggerated facial features and bright colors, standing in a playful pose." \
  -F "position=bottom" \
  -F "scale=1.0" \
  -F "canvas_size=8x10" \
  -F "dpi=300"
```

### With Selfie URL and Background File

```bash
curl -X POST http://localhost:5000/generate-character-web \
  -F "selfie_url=https://example.com/selfie.jpg" \
  -F "background=@/path/to/your/background.jpg" \
  -F "character_prompt=A full-body cartoon caricature with bright colors" \
  -F "position=bottom" \
  -F "scale=1.0"
```

### With Selfie File and Background URL

```bash
curl -X POST http://localhost:5000/generate-character-web \
  -F "selfie=@/path/to/your/selfie.jpg" \
  -F "background_url=https://example.com/background.jpg" \
  -F "character_prompt=A 3D cartoon character in Pixar style wearing colorful streetwear, smiling and posing confidently, full-body." \
  -F "position=bottom" \
  -F "scale=1.0"
```

### With Selfie URL and Background URL

```bash
curl -X POST http://localhost:5000/generate-character-web \
  -F "selfie_url=https://example.com/selfie.jpg" \
  -F "background_url=https://example.com/background.jpg" \
  -F "character_prompt=A cartoon character with vibrant colors" \
  -F "position=bottom" \
  -F "scale=1.0"
```

### With Selfie File Only (No Background)

```bash
curl -X POST http://localhost:5000/generate-character-web \
  -F "selfie=@/path/to/your/selfie.jpg" \
  -F "character_prompt=A heroic fantasy character in detailed digital art style wearing medieval armor, full-body dynamic pose." \
  -F "position=center" \
  -F "scale=1.0"
```

### With Selfie URL Only (No Background)

```bash
curl -X POST http://localhost:5000/generate-character-web \
  -F "selfie_url=https://example.com/selfie.jpg" \
  -F "character_prompt=A cartoon character with vibrant colors" \
  -F "position=center" \
  -F "scale=1.0"
```

### Using API Alias Endpoint

```bash
curl -X POST http://localhost:5000/api/generate-character \
  -F "selfie=@/path/to/your/selfie.jpg" \
  -F "character_prompt=A full-body caricature with bold outlines and vibrant colors." \
  -F "position=bottom" \
  -F "scale=1.0"
```

**Expected Response:**
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

## 3. Background Removal

### With Image File Upload

```bash
curl -X POST http://localhost:5000/remove-bg \
  -F "image=@/path/to/your/image.jpg"
```

### With Image URL

```bash
curl -X POST http://localhost:5000/remove-bg \
  -F "image_url=https://example.com/image.jpg"
```

**Expected Response:**
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

## 4. Download Generated Image (Local)

```bash
curl -X GET http://localhost:5000/download/output_filename.png \
  --output downloaded_image.png
```

## 5. View Generated Image (Local)

```bash
curl -X GET http://localhost:5000/outputs/output_filename.png \
  --output image.png
```

## Complete Example Workflow

### Step 1: Generate Character
```bash
curl -X POST http://localhost:5000/generate-character-web \
  -F "selfie=@selfie.jpg" \
  -F "character_prompt=Cartoon caricature style" \
  -F "position=bottom" \
  -F "scale=1.0" \
  > response.json
```

### Step 2: Extract Image URL from Response
```bash
# On macOS/Linux
cat response.json | grep -o '"image_url":"[^"]*"' | cut -d'"' -f4

# Or use jq (if installed)
cat response.json | jq -r '.image_url'
```

### Step 3: Download from CloudFront
```bash
# If you got the CloudFront URL
curl -X GET "https://d2s4ngnid78ki4.cloudfront.net/converted/uuid_filename.png" \
  --output result.png
```

## Error Responses

### Missing Required Field
```json
{
  "error": "Selfie image is required"
}
```

### Invalid File Type
```json
{
  "error": "Invalid selfie file type"
}
```

### Generation Failed
```json
{
  "error": "Character generation failed: No image generated by Gemini"
}
```

## Notes

- Replace `/path/to/your/` with actual file paths
- Replace example URLs with real image URLs
- The `image_url` field will only be present if S3 upload is successful
- If S3 upload fails, only `local_path` will be available
- Make sure your `.env` file is properly configured with all required variables

## Testing with JSON (Alternative)

If you prefer JSON format, you can use tools like `httpie` or convert form data:

```bash
# Using httpie (install: pip install httpie)
http POST http://localhost:5000/generate-character-web \
  selfie@selfie.jpg \
  character_prompt="Cartoon style" \
  position=bottom \
  scale=1.0
```

