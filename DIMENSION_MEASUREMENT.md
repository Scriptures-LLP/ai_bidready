# Dimension-Based Area Measurement Feature

## Overview
The API now automatically converts shape areas from pixels to **square inches** using detected dimension annotations in floor plans.

## How It Works

### 1. Dimension Detection
The YOLOv8 model detects "Dimension" annotations in the floor plan (e.g., dimension lines with text like "6'- 3 3/4\"").

### 2. Text Extraction
When a Dimension is detected, the API:
- Uses pytesseract OCR to extract the dimension text from the bounding box
- Parses various dimension formats:
  - `6'- 3 3/4"` (feet + inches with fractions)
  - `10'` (feet only)
  - `5' 6"` (feet + inches)
  - `48"` (inches only)
  - `3 1/2"` (inches with fractions)

### 3. Pixel-to-Inch Calibration
The API then:
- Detects the dimension line in the image using Hough line detection
- Measures the pixel length of the dimension line
- Calculates `px_per_inch = pixel_length / real_inches`

### 4. Area Conversion
For each detected shape:
- Original area in pixels² is calculated using OpenCV contours
- If calibration successful: `area_sq_in = area_px / (px_per_inch)²`
- Result is added to the shape object as `area_sq_in`

## API Response

### With Dimension Calibration
```json
{
  "shapes": [
    {
      "path": "M100,200L150,200L150,280L100,280Z",
      "area": 4000.0,
      "area_sq_in": 25.0,
      "color": "#3F5EFB"
    }
  ],
  "dimension_calibration": {
    "text": "6'- 3 3/4\"",
    "px_length": 1212.5,
    "real_inches": 75.75,
    "px_per_inch": 16.0
  }
}
```

### Without Dimension (Fallback)
```json
{
  "shapes": [
    {
      "path": "M100,200L150,200L150,280L100,280Z",
      "area": 4000.0,
      "color": "#3F5EFB"
    }
  ]
}
```

## New Functions in `service/detect.py`

### `parse_dimension_text_to_inches(dim_text: str) -> float`
Parses dimension text and returns total inches.

**Examples:**
- `"6'- 3 3/4\""` → 75.75 inches
- `"10'"` → 120.0 inches
- `"48\""` → 48.0 inches

### `find_horizontal_dimension_length_px(image, bbox, ...) -> float`
Detects the dimension line in the image and returns its pixel length.
Uses Hough line detection with projection fallback.

### `extract_text_from_bbox_ocr(image, bbox) -> str`
Extracts text from a bounding box using pytesseract OCR.

### `compute_px_per_inch_from_dimension(image, bbox, dim_text) -> (float, float, float)`
Returns `(px_per_inch, px_length, real_inches)` from a dimension detection.

### `convert_area_px_to_sqin(area_px: float, px_per_inch: float) -> float`
Converts area from pixels² to square inches.

## Dependencies Added
- `pytesseract` - OCR for dimension text extraction
- `re` - Regular expressions for dimension parsing
- `math` - Mathematical operations

## Installation
```bash
pip install pytesseract
```

Note: You also need to install the Tesseract OCR engine:
- **macOS**: `brew install tesseract`
- **Ubuntu**: `sudo apt-get install tesseract-ocr`
- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki

## Testing

### Unit Tests
```bash
python test_dimension_parsing.py
```

### API Test
```bash
python examples/test_dimension_measurement.py
```

## Supported Dimension Formats
- Feet and inches with fractions: `6'- 3 3/4"`
- Feet and inches: `5' 6"`
- Feet only: `10'`
- Inches with fractions: `3 1/2"`
- Inches only: `48"`
- Decimal inches: `12.5"`

## Error Handling
- If no Dimension annotation is detected, shapes are returned with pixel areas only
- If OCR fails, `dimension_calibration.error` will contain the error message
- If dimension parsing fails, shapes are returned without `area_sq_in`

## Future Enhancements
- Support for vertical dimension lines
- Multiple dimension detection for accuracy
- Metric unit support (cm, m)
- Custom OCR configuration options
