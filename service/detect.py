import cv2
import random
import string
import os
import requests
import tempfile
import io
import numpy as np
import base64
import re
import math
from PIL import Image
try:
    import pytesseract
except ImportError:
    pytesseract = None


def random_string_generator(size=6):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(size))


def _download_image_to_temp(url, suffix=".png", timeout=10):
    """Download `url` to a temporary file and return the file path."""
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
    finally:
        tmp.close()

    return tmp.name


def detect_shapes(image_link, min_area=800, max_area=100000, colorize: bool = False):
    """Detect polygon path data for contours in an image.

    `image_link` may be a local filesystem path or an HTTP/HTTPS URL. If it's a
    URL the image is downloaded to a temporary file, processed, and then the
    temporary file is removed.

    Returns a list of objects for each path in the form:
        {'path': 'M10,10L20,10L20,20Z', 'area': 1234.5}

    If `colorize=True`, a 'color' key will also be added:
        {'path': '...', 'area': 1234.5, 'color': '#aabbcc'}
    """
    downloaded_temp = None
    # Determine whether image_link is a URL
    is_url = isinstance(image_link, str) and image_link.lower().startswith(("http://", "https://"))

    try:
        if is_url:
            # Choose a reasonable suffix; try to preserve extension if present
            _, ext = os.path.splitext(image_link)
            suffix = ext if ext and len(ext) <= 5 else ".png"
            downloaded_temp = _download_image_to_temp(image_link, suffix=suffix)
            image_path = downloaded_temp
        else:
            image_path = image_link

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        (image_height, image_width) = image.shape[:2]

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Otsu thresholding
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize a list to store the detected shapes
        detected_plots_position = []

        for i, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)
            # Check if the contour area is within the specified range
            if min_area <= contour_area <= max_area:
                path_data = "M" + "L".join([f"{point[0][0]},{point[0][1]}" for point in contour]) + "Z"
                # compute contour area
                contour_area = float(contour_area)
                if colorize:
                    # Assign a random color per path (hex)
                    col = '#' + ''.join(random.choice('0123456789ABCDEF') for _ in range(6))
                    detected_plots_position.append({
                        'path': path_data,
                        'area': contour_area,
                        'color': col,
                    })
                else:
                    detected_plots_position.append({
                        'path': path_data,
                        'area': contour_area,
                    })

        return detected_plots_position

    finally:
        # Clean up the downloaded temporary file (if any)
        if downloaded_temp and os.path.exists(downloaded_temp):
            try:
                os.remove(downloaded_temp)
            except Exception:
                pass


def _load_image_bytes_and_size(image_link, timeout=10):
        """Load the image (URL or local path) and return tuple (bytes, width, height, mime_ext).

        mime_ext is the extension string like 'png' or 'jpeg' and used for Data URI creation.
        """
        # Determine URL or local
        is_url = isinstance(image_link, str) and image_link.lower().startswith(("http://", "https://"))
        img_bytes = None
        if is_url:
                resp = requests.get(image_link, stream=True, timeout=timeout)
                resp.raise_for_status()
                img_bytes = resp.content
        else:
                # read local file
                with open(image_link, 'rb') as f:
                        img_bytes = f.read()

        # Determine extension (mime type) by inspecting bytes via PIL
        try:
                img = Image.open(io.BytesIO(img_bytes))
                width, height = img.size
                fmt = img.format.lower() if img.format else 'png'
        except Exception:
                # fallback: use opencv to get dimensions
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_cv = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                if img_cv is None:
                        raise ValueError("Could not decode image to determine size")
                height, width = img_cv.shape[:2]
                fmt = 'png'

        return img_bytes, width, height, fmt


def _data_uri_from_bytes(bytes_data, fmt):
        """Return a data URI string for the bytes and format (png, jpeg, etc.)."""
        b64 = base64.b64encode(bytes_data).decode('ascii')
        mime = 'jpeg' if fmt in ('jpg', 'jpeg') else fmt
        return f"data:image/{mime};base64,{b64}"


def build_svg_from_paths(
        paths,
        width,
        height,
        stroke_color="#ff0000",
        stroke_width=2,
        svg_fill="none",
        fill_opacity: float = 0.12,
    ):
        """Build an SVG string from paths returned by detect_shapes.

        - `paths` may be a list of strings or a list of dicts {'path':..., 'color':...}.
        - Returns only the SVG element string (including <svg> wrapper).
        """
        svg_paths = []
        for p in paths:
            if isinstance(p, dict):
                path_str = p.get('path')
                col = p.get('color', stroke_color)
                area_val = p.get('area')
            else:
                path_str = p
                col = stroke_color
                area_val = None

            if svg_fill and svg_fill.strip().lower() != "none":
                fill_value = svg_fill
                fill_op = fill_opacity
            else:
                if isinstance(p, dict):
                    fill_value = col
                    fill_op = fill_opacity
                else:
                    if svg_fill and svg_fill.strip().lower() == "none":
                        fill_value = "none"
                        fill_op = 1.0
                    else:
                        fill_value = svg_fill
                        fill_op = fill_opacity

            # Add a data-area attribute for the path (if area known) and a title for hover
            area_attr = f' data-area="{area_val}"' if area_val is not None else ''
            attrs = (
                f'stroke="{col}" stroke-width="{stroke_width}" '
                f'fill="{fill_value}" fill-opacity="{fill_op}" stroke-linejoin="round" stroke-linecap="round"{area_attr}'
            )
            if area_val is not None:
                svg_paths.append(f'<path d="{path_str}" {attrs}><title>{area_val}</title></path>')
            else:
                svg_paths.append(f'<path d="{path_str}" {attrs} />')

        svg_inner = "\n".join(svg_paths)
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">\n{svg_inner}\n</svg>'
        return svg


def detect_shapes_html(
        image_link,
        min_area=1000,
        max_area=90000,
        stroke_color="#ff0000",
        stroke_width=2,
    svg_fill="none",
    fill_opacity: float = 0.12,
        colorize=False,
        mode="general",
        timeout=10,
):
        """Generate an HTML string containing the original image overlaid with SVG paths.

        - Calls `detect_shapes` (unchanged) to detect polygon path data.
        - Embeds the image as a data URI (so HTML is standalone).
        - Builds an SVG overlay sized to the image size and places the detected paths.

        Returns an HTML string.
        """
        # Collect shapes from the existing function, ask it to colorize if needed
        paths = detect_shapes(image_link, min_area=min_area, max_area=max_area, colorize=colorize)

        # Load image bytes and size
        try:
                bytes_data, width, height, fmt = _load_image_bytes_and_size(image_link, timeout=timeout)
        except Exception as e:
                raise ValueError(f"Could not load image for HTML generation: {str(e)}")

        data_uri = _data_uri_from_bytes(bytes_data, fmt)

        # Build SVG using the helper
        svg = build_svg_from_paths(
            paths,
            width,
            height,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            svg_fill=svg_fill,
            fill_opacity=fill_opacity,
        )

        # Build a simple HTML with the image and the svg overlay absolutely positioned
        html = f'''<!doctype html>
<html>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Image with SVG overlay</title>
        <style>
            .overlay-container {{
                position: relative;
                display: inline-block;
            }}
            .overlay-container img {{
                display: block;
                width: {width}px;
                height: {height}px;
            }}
            .overlay-container svg {{
                position: absolute;
                left: 0;
                top: 0;
                width: {width}px;
                height: {height}px;
                pointer-events: none;
            }}
        </style>
    </head>
    <body>
        <div class="overlay-container">
            <img src="{data_uri}" alt="Image" />
            {svg}
        </div>
    </body>
</html>'''

        return html


def parse_dimension_text_to_inches(dim_text: str) -> float:
    """Parse dimension strings like `6'- 3 3/4\"`, `6'3.75\"`, `75\"`, `6 ft 3 3/4 in`
    and return total inches (float)."""
    if not dim_text or not isinstance(dim_text, str):
        raise ValueError("No dimension text provided")

    s = dim_text.replace("\u2019", "'").replace("\u2033", '"').replace("\u201d", '"').replace("\u2032","'").strip()
    feet = 0
    inches = 0.0

    m_feet = re.search(r"(?P<ft>\d+)\s*'", s)
    if m_feet:
        feet = int(m_feet.group("ft"))
        rest = s[m_feet.end():]
    else:
        rest = s

    # whole inches + fraction e.g., 3 3/4"
    m_inch_whole_frac = re.search(r"(?P<iw>\d+)\s*(?:-| )?\s*(?P<num>\d+)/(?P<den>\d+)", rest)
    if m_inch_whole_frac:
        inches = int(m_inch_whole_frac.group("iw")) + (int(m_inch_whole_frac.group("num")) / int(m_inch_whole_frac.group("den")))
    else:
        # fraction only
        m_frac_only = re.search(r"(?P<num>\d+)/(?P<den>\d+)", rest)
        if m_frac_only:
            inches = int(m_frac_only.group("num")) / int(m_frac_only.group("den"))
        else:
            # decimal or whole inches
            m_inches = re.search(r"(?P<in>\d+(?:\.\d+)?)\s*(?:\"|in|inch|inches)?", rest)
            if m_inches:
                inches = float(m_inches.group("in"))

    total_inches = feet * 12.0 + inches
    if total_inches <= 0:
        raise ValueError(f"Could not parse dimension value from '{dim_text}'")
    return total_inches


def _load_cv2_image(image_path_or_bytes):
    """Helper to load cv image from local path or bytes."""
    if isinstance(image_path_or_bytes, (bytes, bytearray)):
        nparr = np.frombuffer(image_path_or_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        img_cv = cv2.imread(image_path_or_bytes)
    if img_cv is None:
        raise ValueError("Cannot load image for measurement")
    return img_cv


def find_horizontal_dimension_length_px(image_path_or_bytes, bbox, search_half_height: int = 12, hough_min_length: int = 30):
    """Given image and bbox (dict/x1,y1,x2,y2), find horizontal dimension line near bbox center
    and return pixel length. Uses Hough detection with fallback to projection."""
    img_cv = _load_cv2_image(image_path_or_bytes)
    h, w = img_cv.shape[:2]

    x1, y1, x2, y2 = int(round(bbox['x1'])), int(round(bbox['y1'])), int(round(bbox['x2'])), int(round(bbox['y2']))
    center_y = int(round((y1 + y2) / 2.0))
    center_x = int(round((x1 + x2) / 2.0))

    # Prepare edges for full image Hough
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges_full = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges_full, 1, np.pi / 180, threshold=120, minLineLength=hough_min_length, maxLineGap=10)

    candidate_lines = []
    if lines is not None:
        for line in lines.reshape(-1, 4):
            x_start, y_start, x_end, y_end = line
            # Check near horizontal
            if abs(y_start - y_end) <= max(2, int(0.01 * h)):
                y_mid = (y_start + y_end) / 2.0
                if abs(y_mid - center_y) <= max(10, int(0.02 * h)):
                    candidate_lines.append(((x_start, y_start, x_end, y_end), math.hypot(x_end - x_start, y_end - y_start)))

    if candidate_lines:
        best_line = max(candidate_lines, key=lambda x: x[1])[0]
        px_len = math.hypot(best_line[2] - best_line[0], best_line[3] - best_line[1])
        return float(px_len)

    # Fallback: projection on strip
    ymin = max(0, center_y - search_half_height)
    ymax = min(h - 1, center_y + search_half_height)
    strip = img_cv[ymin:ymax+1, :]
    gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    _, thr_strip = cv2.threshold(gray_strip, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    column_sum = np.sum(thr_strip > 0, axis=0)
    col_mask = column_sum > max(1, 0.02 * thr_strip.shape[0])

    segments = []
    start = None
    for i, val in enumerate(col_mask):
        if val:
            if start is None:
                start = i
        else:
            if start is not None:
                segments.append((start, i - 1))
                start = None
    if start is not None:
        segments.append((start, len(col_mask) - 1))

    if not segments:
        return float(abs(x2 - x1))

    window_x = center_x
    best_seg = None
    best_dist = None
    for seg in segments:
        seg_cx = (seg[0] + seg[1]) // 2
        dist = abs(seg_cx - window_x)
        if best_seg is None or dist < best_dist:
            best_seg = seg
            best_dist = dist

    if best_seg:
        seg_start, seg_end = best_seg
        return float(seg_end - seg_start)

    return float(abs(x2 - x1))


def extract_text_from_bbox_ocr(image_path_or_bytes, bbox):
    """Extract text from a bounding box using pytesseract OCR.
    bbox: dict with x1,y1,x2,y2 keys.
    Returns the extracted text string (stripped)."""
    if pytesseract is None:
        raise ImportError("pytesseract is not installed. Install with: pip install pytesseract")
    
    img_cv = _load_cv2_image(image_path_or_bytes)
    x1, y1, x2, y2 = int(round(bbox['x1'])), int(round(bbox['y1'])), int(round(bbox['x2'])), int(round(bbox['y2']))
    
    # Crop the region
    cropped = img_cv[y1:y2, x1:x2]
    
    # Convert to PIL for pytesseract
    cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    
    # Extract text with basic config
    text = pytesseract.image_to_string(cropped_pil, config='--psm 7')  # psm 7: single line
    return text.strip()


def compute_px_per_inch_from_dimension(image_path_or_bytes, bbox, dim_text):
    """Returns (px_per_inch, px_length_px, real_inches)
    - bbox: dict with x1,y1,x2,y2 of the detected Dimension annotation
    - dim_text: detected dimension string (e.g., "6'- 3 3/4\"")
    """
    real_inches = parse_dimension_text_to_inches(dim_text)
    px_length = find_horizontal_dimension_length_px(image_path_or_bytes, bbox)
    if px_length <= 0:
        raise ValueError("Could not determine pixel length for the detected dimension")
    px_per_inch = float(px_length) / float(real_inches)
    return px_per_inch, px_length, real_inches


def convert_area_px_to_sqin(area_px: float, px_per_inch: float) -> float:
    """Convert area from px^2 to square inches using px_per_inch."""
    if px_per_inch <= 0:
        raise ValueError("px_per_inch must be > 0")
    return float(area_px) / (px_per_inch ** 2)


def convert_area_px_to_sqft(area_px: float, px_per_inch: float) -> float:
    """Convert px^2 to square feet: convert to sq in then divide by 144."""
    sq_in = convert_area_px_to_sqin(area_px, px_per_inch)
    return sq_in / 144.0

