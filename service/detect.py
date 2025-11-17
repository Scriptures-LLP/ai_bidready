import cv2
import random
import string
import os
import requests
import tempfile
import io
import numpy as np
import base64
from PIL import Image


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

    Returns a list of path strings (e.g. 'M10,10L20,10L20,20Z').
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
                if colorize:
                    # Assign a random color per path (hex)
                    col = '#' + ''.join(random.choice('0123456789ABCDEF') for _ in range(6))
                    detected_plots_position.append({
                        'path': path_data,
                        'color': col,
                    })
                else:
                    detected_plots_position.append(path_data)

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
            else:
                path_str = p
                col = stroke_color

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

            attrs = (
                f'stroke="{col}" stroke-width="{stroke_width}" '
                f'fill="{fill_value}" fill-opacity="{fill_op}" stroke-linejoin="round" stroke-linecap="round"'
            )
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
