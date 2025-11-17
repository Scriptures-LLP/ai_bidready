import cv2
import random
import string
import os
import requests
import tempfile


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


def detect_shapes(image_link, min_area=800, max_area=100000):
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
                detected_plots_position.append(path_data)

        return detected_plots_position

    finally:
        # Clean up the downloaded temporary file (if any)
        if downloaded_temp and os.path.exists(downloaded_temp):
            try:
                os.remove(downloaded_temp)
            except Exception:
                pass
