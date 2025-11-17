#!/usr/bin/env python3
"""
Example: Test dimension-based area measurement
This demonstrates how the API automatically converts shape areas from pixels to square inches
using detected dimension annotations.
"""

import requests
import json

# API endpoint
API_URL = "http://localhost:8000/detect"

# Example floor plan with dimension annotation
# Replace with your actual image URL or local path
IMAGE_URL = "path/to/floorplan_with_dimension.jpg"

def test_dimension_measurement():
    """Test the /detect endpoint with dimension-based measurement."""
    
    print("Testing dimension-based area measurement...")
    print(f"Image: {IMAGE_URL}")
    print("-" * 80)
    
    # Prepare request
    payload = {
        "image_url": IMAGE_URL,
        "confidence": 0.25,
        "selected_labels": None,  # detect all labels
        "use_tiling": None  # auto-detect
    }
    
    try:
        # Send request
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        # Display results
        print(f"✓ Success! Total detections: {data['total_detections']}")
        print(f"  Object counts: {data['object_counts']}")
        print()
        
        # Check if dimension calibration was performed
        if "dimension_calibration" in data:
            cal = data["dimension_calibration"]
            if "error" in cal:
                print(f"⚠️  Dimension calibration failed: {cal['error']}")
            else:
                print("📐 Dimension Calibration:")
                print(f"   Text detected: {cal['text']}")
                print(f"   Pixel length: {cal['px_length']:.2f} px")
                print(f"   Real length: {cal['real_inches']:.2f} inches")
                print(f"   Ratio: {cal['px_per_inch']:.4f} px/inch")
                print()
        else:
            print("⚠️  No Dimension annotation detected - areas in pixels only")
            print()
        
        # Display shape measurements
        if "shapes" in data and data["shapes"]:
            print(f"📏 Shape Measurements ({len(data['shapes'])} shapes):")
            print("-" * 80)
            for i, shape in enumerate(data["shapes"][:5], 1):  # show first 5
                area_px = shape.get("area", 0)
                area_sqin = shape.get("area_sq_in")
                
                if area_sqin:
                    area_sqft = area_sqin / 144.0
                    print(f"   Shape {i}:")
                    print(f"      Area (pixels): {area_px:.2f} px²")
                    print(f"      Area (inches): {area_sqin:.2f} sq in")
                    print(f"      Area (feet):   {area_sqft:.4f} sq ft")
                else:
                    print(f"   Shape {i}:")
                    print(f"      Area (pixels): {area_px:.2f} px²")
                    print(f"      (No dimension calibration - cannot convert to real units)")
                print()
            
            if len(data["shapes"]) > 5:
                print(f"   ... and {len(data['shapes']) - 5} more shapes")
        else:
            print("   No shapes detected")
        
        print()
        print("✓ Test completed successfully!")
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    print("=" * 80)
    print("Dimension-Based Area Measurement Test")
    print("=" * 80)
    print()
    
    test_dimension_measurement()
