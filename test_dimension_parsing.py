#!/usr/bin/env python3
"""Test dimension parsing and area conversion functions."""

from service.detect import parse_dimension_text_to_inches, convert_area_px_to_sqin

def test_parse_dimension_text():
    """Test various dimension text formats."""
    test_cases = [
        ("6'- 3 3/4\"", 75.75),  # 6 feet + 3.75 inches = 75.75 inches
        ("6' 3 3/4\"", 75.75),
        ("6'-3 3/4\"", 75.75),
        ("10'", 120.0),  # 10 feet = 120 inches
        ("5' 6\"", 66.0),  # 5 feet + 6 inches = 66 inches
        ("48\"", 48.0),  # 48 inches
        ("3 1/2\"", 3.5),  # 3.5 inches
        ("1/2\"", 0.5),  # 0.5 inches
        ("12.5\"", 12.5),  # 12.5 inches (decimal)
    ]
    
    print("Testing dimension text parsing:")
    print("-" * 60)
    for dim_text, expected in test_cases:
        try:
            result = parse_dimension_text_to_inches(dim_text)
            status = "✓" if abs(result - expected) < 0.01 else "✗"
            print(f"{status} '{dim_text}' -> {result:.2f} inches (expected {expected:.2f})")
        except Exception as e:
            print(f"✗ '{dim_text}' -> ERROR: {e}")
    print()

def test_area_conversion():
    """Test area conversion from pixels to square inches."""
    print("Testing area conversion:")
    print("-" * 60)
    
    # Example: if px_per_inch = 10, then 1 square inch = 100 px²
    px_per_inch = 10.0
    test_cases = [
        (100, 1.0),    # 100 px² = 1 sq in
        (400, 4.0),    # 400 px² = 4 sq in
        (1000, 10.0),  # 1000 px² = 10 sq in
    ]
    
    for area_px, expected_sqin in test_cases:
        result = convert_area_px_to_sqin(area_px, px_per_inch)
        status = "✓" if abs(result - expected_sqin) < 0.01 else "✗"
        print(f"{status} {area_px} px² @ {px_per_inch} px/in -> {result:.2f} sq in (expected {expected_sqin:.2f})")
    print()

if __name__ == "__main__":
    test_parse_dimension_text()
    test_area_conversion()
    print("All tests completed!")
