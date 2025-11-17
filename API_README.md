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
- **Content-Type:** `application/json` or `multipart/form-data`
- **Parameters (JSON body):**
  ```json
  {
    "image_url": "http://example.com/floorplan.jpg",
    "confidence": 0.25,
    "selected_labels": "Door,Window,Wall",
    "use_tiling": null
  }
  ```
- **Parameters (form-data):**
  - `image_url`: URL or local file path to the floor plan image
  - `confidence`: Detection threshold (0.0-1.0, default: 0.25)
  - `selected_labels`: Comma-separated list of labels to detect (optional)
  - `use_tiling`: Enable/disable tiling for large images (optional, auto-detected if null)

**New Feature:** Automatic area measurement in square inches!
- If the floor plan includes a "Dimension" annotation (e.g., "6'-3 3/4\""), the API will:
  1. Detect the dimension text using OCR
  2. Calculate the pixel-to-inch ratio from the dimension line
  3. Convert all shape areas from pixels² to square inches
  4. Return `area_sq_in` in each shape object

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
  "parameters": {
    "confidence": 0.25,
    "selected_labels": ["Door", "Window", "Wall"],
    "tiling_used": false,
    "original_image_size": {"width": 1920, "height": 1080}
  },
  "shapes": [
    {
      "path": "M100,200L150,200L150,280L100,280Z",
      "area": 4000.0,
      "area_sq_in": 25.0,
      "color": "#3F5EFB"
    }
  ],
  "shapes_svg": "<svg xmlns=\"http://www.w3.org/2000/svg\" ...>...</svg>",
  "dimension_calibration": {
    "text": "6'- 3 3/4\"",
    "px_length": 1212.5,
    "real_inches": 75.75,
    "px_per_inch": 16.0
  }
}
```

**New Fields:**
- `shapes`: Array of detected contour shapes with SVG paths, pixel area, and **square inch area** (if dimension found)
- `shapes_svg`: SVG overlay string for visualizing shapes
- `dimension_calibration`: Calibration data from the first detected Dimension annotation (if any)
  - `text`: OCR-extracted dimension text (e.g., "6'- 3 3/4\"")
  - `px_length`: Measured pixel length of the dimension line
  - `real_inches`: Parsed real-world length in inches
  - `px_per_inch`: Calculated pixel-per-inch ratio used for area conversion

## Usage Examples

### JavaScript/Fetch API
```javascript
// Using JSON body with image URL
fetch('http://localhost:8000/detect', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    image_url: 'http://example.com/floorplan.jpg',
    confidence: 0.5,
    selected_labels: 'Door,Window,Wall'
  })
})
.then(response => response.json())
.then(data => {
  console.log('Detections:', data.detections);
  console.log('Shapes with areas in sq inches:', data.shapes);
  console.log('Calibration:', data.dimension_calibration);
});

// Using form-data with image URL
const formData = new FormData();
formData.append('image_url', 'http://example.com/floorplan.jpg');
formData.append('confidence', '0.5');
formData.append('selected_labels', 'Door,Window');

fetch('http://localhost:8000/detect', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

### curl
```bash
# JSON request with image URL
curl -X POST "http://localhost:8000/detect" \
     -H "Content-Type: application/json" \
     -d '{
       "image_url": "http://example.com/floorplan.jpg",
       "confidence": 0.5,
       "selected_labels": "Door,Window,Wall"
     }'

# Form-data request with local file path
curl -X POST "http://localhost:8000/detect" \
     -F "image_url=/path/to/floorplan.jpg" \
     -F "confidence=0.5" \
     -F "selected_labels=Door,Window"

# Simple detection (file upload)
curl -X POST "http://localhost:8000/detect-simple" \
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