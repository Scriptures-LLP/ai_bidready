#!/usr/bin/env python3
"""
Complete example: Dimension-based measurement workflow
This shows the full process from dimension detection to area conversion.
"""

from service.detect import (
    parse_dimension_text_to_inches,
    find_horizontal_dimension_length_px,
    compute_px_per_inch_from_dimension,
    convert_area_px_to_sqin,
    detect_shapes
)

# Example 1: Parse dimension text
print("=" * 80)
print("Example 1: Dimension Text Parsing")
print("=" * 80)

dimension_texts = [
    "6'- 3 3/4\"",
    "10'",
    "5' 6\"",
    "48\"",
    "3 1/2\"",
]

for dim_text in dimension_texts:
    inches = parse_dimension_text_to_inches(dim_text)
    feet = inches / 12.0
    print(f"'{dim_text:15}' = {inches:6.2f} inches = {feet:5.2f} feet")

print()

# Example 2: Area conversion
print("=" * 80)
print("Example 2: Area Conversion")
print("=" * 80)

# Assume we measured a dimension line:
# - The dimension text says "10'" (10 feet = 120 inches)
# - The pixel length of the line is 1200 pixels
# - Therefore: px_per_inch = 1200 / 120 = 10.0

px_per_inch = 10.0
print(f"Calibration: {px_per_inch} pixels per inch")
print()

# Now convert some example areas
example_areas_px = [
    ("Small room", 10000),
    ("Medium room", 50000),
    ("Large room", 100000),
]

for name, area_px in example_areas_px:
    area_sqin = convert_area_px_to_sqin(area_px, px_per_inch)
    area_sqft = area_sqin / 144.0
    print(f"{name:15}: {area_px:7.0f} px² = {area_sqin:7.2f} sq in = {area_sqft:6.2f} sq ft")

print()

# Example 3: Complete workflow (requires actual image with dimension)
print("=" * 80)
print("Example 3: Complete Workflow")
print("=" * 80)
print("""
For a complete workflow with an actual floor plan image:

1. Detect dimension annotation with YOLOv8
   → Returns bbox: {x1, y1, x2, y2}

2. Extract dimension text using OCR
   → extract_text_from_bbox_ocr(image, bbox)
   → Returns: "6'- 3 3/4\""

3. Compute px_per_inch calibration
   → px_per_inch, px_len, real_in = compute_px_per_inch_from_dimension(image, bbox, text)
   → Returns: (16.0, 1212.0, 75.75)

4. Detect shapes in the image
   → shapes = detect_shapes(image_url, colorize=True)
   → Returns: [{'path': '...', 'area': 4000.0, 'color': '#...'}]

5. Convert all shape areas to square inches
   → for shape in shapes:
   →     shape['area_sq_in'] = convert_area_px_to_sqin(shape['area'], px_per_inch)

6. Return results with real-world measurements
   → API response includes both pixel and square inch areas

This is now automated in the /detect endpoint!
""")

print()
print("✓ Examples completed!")
print()
print("To test with a real floor plan:")
print("  1. Make sure the floor plan has a dimension annotation (e.g., '6'-3 3/4\"')")
print("  2. Run: python examples/test_dimension_measurement.py")
print("  3. Or POST to /detect endpoint with image_url parameter")
