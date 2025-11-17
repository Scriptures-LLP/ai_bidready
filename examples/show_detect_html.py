"""Example script to call detect_shapes_html and save the generated HTML.

Usage:
    python examples/show_detect_html.py <image_url_or_path> [output.html]

If `output.html` is not provided the script writes `out.html` in the current directory.
"""
import argparse
import os
import sys

# Ensure project root is on sys.path so `service` package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from service.detect import detect_shapes_html


def main():
    parser = argparse.ArgumentParser(description="Generate HTML with SVG overlay from an image URL or path")
    parser.add_argument("image", help="Image URL (http/https) or local path")
    parser.add_argument("output", nargs="?", default="out.html", help="Output HTML file (default: out.html)")
    parser.add_argument("--min-area", type=int, default=800, help="Minimum contour area to keep")
    parser.add_argument("--max-area", type=int, default=100000, help="Maximum contour area to keep")
    parser.add_argument("--stroke-color", default="#ff0000", help="SVG stroke color")
    parser.add_argument("--stroke-width", type=int, default=2, help="SVG stroke width")
    parser.add_argument("--fill", default="none", help="SVG fill for paths (default 'none')")
    parser.add_argument("--fill-opacity", type=float, default=0.12, help="Fill opacity (0.0-1.0) for paths; ignored if fill='none' (default 0.12)")
    parser.add_argument("--colorize", action="store_true", help="Colorize detected shapes (per-shape colors)")
    parser.add_argument("--mode", choices=["rooms", "corridors", "general"], default="general", help="Processing mode (default: general)")
    args = parser.parse_args()

    html = detect_shapes_html(
        args.image,
        min_area=1000,
        max_area=90000,
        stroke_color=args.stroke_color,
        stroke_width=args.stroke_width,
        svg_fill=args.fill,
        fill_opacity=args.fill_opacity,
        colorize=args.colorize,
        mode=args.mode,
    )

    out_path = args.output
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved HTML with overlay to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
