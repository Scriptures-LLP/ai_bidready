# BidReady AI Model API

## Overview
A REST API for floor plan object detection using YOLOv8. This API wraps the AI model functionality to allow integration with websites and applications.

## API Endpoints

### Base URL
- **Local:** `http://localhost:8000`
- **Network:** `http://192.168.1.38:8000`

### Available Endpoints

#### 1. Root Information
- **GET** `/`
- Returns API information and available endpoints

#### 2. Health Check  
- **GET** `/health`
- Check if the API and model are running properly

#### 3. Available Labels
- **GET** `/labels`
- Get list of all detectable object types

#### 4. Object Detection (Full)
- **POST** `/detect`
- **Parameters:**
  - `file`: Image file (JPG, JPEG, PNG)
  - `confidence`: Detection threshold (0.0-1.0, default: 0.25)
  - `selected_labels`: Comma-separated list of labels to detect (optional)

#### 5. Object Detection (Simple)
- **POST** `/detect-simple`  
- **Parameters:**
  - `file`: Image file (JPG, JPEG, PNG)
- Uses default settings (confidence=0.25, all labels)

## Response Format

### Detection Response
```json
{
  "success": true,
  "total_detections": 5,
  "object_counts": {
    "Door": 2,
    "Window": 3
  },
  "detections": [
    {
      "label": "Door",
      "confidence": 0.85,
      "bbox": {
        "x1": 100.5,
        "y1": 200.3,
        "x2": 150.8,
        "y2": 280.1
      }
    }
  ],
  "annotated_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "parameters": {
    "confidence": 0.25,
    "selected_labels": ["Door", "Window", "Wall"]
  }
}
```

## Usage Examples

### JavaScript/Fetch API
```javascript
// Simple detection
const formData = new FormData();
formData.append('file', imageFile);

fetch('http://localhost:8000/detect-simple', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));

// Advanced detection with parameters
const formData2 = new FormData();
formData2.append('file', imageFile);

fetch('http://localhost:8000/detect?confidence=0.5&selected_labels=Door,Window', {
  method: 'POST',
  body: formData2
})
.then(response => response.json())
.then(data => console.log(data));
```

### curl
```bash
# Simple detection
curl -X POST "http://localhost:8000/detect-simple" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"

# With parameters
curl -X POST "http://localhost:8000/detect?confidence=0.5&selected_labels=Door,Window" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

## Detectable Objects
- Column
- Curtain Wall  
- Dimension
- Door
- Railing
- Sliding Door
- Stair Case
- Wall
- Window

## Interactive Documentation
Visit `http://localhost:8000/docs` for interactive Swagger UI documentation where you can test all endpoints directly in your browser.

## Running the API

### Start the Server
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Stop the Server
Press `Ctrl+C` in the terminal

## CORS Configuration
The API is configured to accept requests from any origin (`*`). For production use, update the `allow_origins` in `api.py` to include only your website's domain.

## Error Handling
- **400**: Bad request (invalid file type, parameters)
- **500**: Internal server error (model loading, processing failures)

All errors return JSON with detailed error messages.