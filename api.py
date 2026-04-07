import base64
import io
import os
import tempfile
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import cv2
import numpy as np
import PIL.Image
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import ImageDraw
from service.detect import (
    build_svg_from_paths,
    compute_actual_sqft_from_drawing,
    compute_px_per_inch_from_dimension,
    convert_area_px_to_sqin,
    detect_shapes,
    extract_text_from_bbox_rekognition,
    parse_scale_text,
)
from ultralytics import YOLO

app = FastAPI(
    title="BidReady AI Model API",
    description="Floor Plan Object Detection API using YOLOv8",
    version="1.0.0",
)

# Add CORS middleware to allow requests from your website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your website's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None


def is_false_positive_wall(bbox, img_w, img_h):
    """
    Check if a detected wall is likely a false positive (e.g., border, grid line, title block).
    """
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return True

    # 1. Detect border lines (page frames wrapping the drawing)
    # They usually sit extremely close to the edges (within 3%) and span almost the entire page (> 85%)
    aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 9999
    
    edge_thresh_x = max(20, img_w * 0.03)  
    edge_thresh_y = max(20, img_h * 0.03)  

    near_top = (y1 < edge_thresh_y)
    near_bottom = (y2 > img_h - edge_thresh_y)
    near_left = (x1 < edge_thresh_x)
    near_right = (x2 > img_w - edge_thresh_x)
    in_title_block = (x1 > img_w * 0.85)
    edge_or_title = near_top or near_bottom or near_left or near_right or in_title_block

    # Horizontal page borders
    if w > (img_w * 0.85) and (near_top or near_bottom) and aspect > 10:
        return True

    # Vertical page borders
    if h > (img_h * 0.85) and (near_left or near_right) and aspect > 10:
        return True

    # CASE C: Large Frame detection
    # If a single "wall" covers > 60% of the image area
    box_area = w * h
    img_area = img_w * img_h
    if box_area > (img_area * 0.6):
        return True

    # CASE D: Large near-edge boxes (likely title block or grid artifacts)
    # Only apply aggressive filtering near borders/title block to protect interior walls.
    if edge_or_title:
        if box_area > (img_area * 0.02) and aspect < 4:
            return True
        if box_area > (img_area * 0.05) and aspect < 8:
            return True

    # CASE E: Dimension/grid lines (long thin lines) near edges or title block
    if edge_or_title and aspect > 60:
        return True

    return False


def load_model():
    """Load the YOLO model with proper torch configuration"""
    global model
    if model is None:
        # Temporarily patch torch.load to use weights_only=False
        original_torch_load = torch.load

        def patched_torch_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        torch.load = patched_torch_load
        try:
            model = YOLO("best.pt")
        finally:
            torch.load = original_torch_load
    return model


def create_tiles(
    image: PIL.Image.Image, tile_size: int, overlap: int
) -> List[Tuple[PIL.Image.Image, Tuple[int, int]]]:
    """
    Split large image into overlapping tiles for better detection

    Args:
        image: Input PIL Image
        tile_size: Size of each tile (default 1920x1920)
        overlap: Overlap between tiles in pixels (default 320)

    Returns:
        List of tuples (tile_image, (x_offset, y_offset))
    """
    width, height = image.size
    tiles = []

    # Calculate step size (tile_size - overlap)
    step = tile_size - overlap

    for y in range(0, height, step):
        for x in range(0, width, step):
            # Calculate tile boundaries
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)

            # Adjust start position if we're at the edge
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)

            # Crop tile
            tile = image.crop((x_start, y_start, x_end, y_end))
            tiles.append((tile, (x_start, y_start)))

    return tiles


def merge_detections(
    all_detections: List, image_size: Tuple[int, int], iou_threshold: float = 0.5
):
    """
    Merge detections from multiple tiles, removing duplicates using NMS

    Args:
        all_detections: List of detection dictionaries
        image_size: Original image size (width, height)
        iou_threshold: IoU threshold for NMS

    Returns:
        Filtered list of detections
    """
    if not all_detections:
        return []

    # Convert to numpy arrays for NMS (we'll build xyxy and convert to xywh for cv2 NMSBoxes)
    boxes_xyxy = []
    scores = []
    labels = []

    for det in all_detections:
        bbox = det["bbox"]
        boxes_xyxy.append([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
        scores.append(det["confidence"])
        labels.append(det["label"])

    boxes_xyxy = np.array(boxes_xyxy, dtype=float)
    scores = np.array(scores)

    # Apply NMS per class
    keep_indices = []
    unique_labels = set(labels)

    for label in unique_labels:
        label_mask = np.array([l == label for l in labels])
        label_boxes_xyxy = boxes_xyxy[label_mask]
        label_scores = scores[label_mask]
        label_indices = np.where(label_mask)[0]

        if len(label_boxes_xyxy) > 0:
            # Convert xyxy -> xywh as required by cv2.dnn.NMSBoxes
            label_boxes_xywh = []
            for x1, y1, x2, y2 in label_boxes_xyxy.tolist():
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                label_boxes_xywh.append([float(x1), float(y1), float(w), float(h)])

            # Apply NMS using cv2 with a low score threshold (actual filtering handled by model confidence)
            indices = cv2.dnn.NMSBoxes(
                bboxes=label_boxes_xywh,
                scores=label_scores.astype(float).tolist(),
                score_threshold=1e-6,
                nms_threshold=float(iou_threshold),
            )

            if len(indices) > 0:
                keep_indices.extend(label_indices[np.array(indices).flatten()].tolist())

    # Return filtered detections
    return [all_detections[i] for i in keep_indices]


def should_use_tiling(image: PIL.Image.Image, threshold: int = 2000) -> bool:
    """
    Determine if image should be processed with tiling

    Args:
        image: Input PIL Image
        threshold: Pixel threshold for largest dimension

    Returns:
        True if tiling should be used
    """
    width, height = image.size
    return max(width, height) > threshold


async def get_building_mask_from_gemini(image: PIL.Image.Image):
    """
    Sends the blueprint to Gemini 1.5 Flash / 3.0 Preview to dynamically validate
    where the true architectural structure is, skipping outside grid tracking.
    """
    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyCDQ0PeW3GKx6NK2PiHViRFBnvQJGPU1ck") # Use env or fallback
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={api_key}"
    
    max_size = 1000
    w, h = image.size
    scale = min(max_size / w, max_size / h)
    if scale < 1.0:
        img_resized = image.resize((int(w * scale), int(h * scale)))
    else:
        img_resized = image
        
    buf = io.BytesIO()
    img_resized.convert("RGB").save(buf, format="JPEG", quality=80)
    b64_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    payload = {
        "contents": [{
            "parts": [
                {"text": "Analyze this architectural blueprint. Return a JSON object with a single key 'building_bbox' containing [ymin, xmin, ymax, xmax] coordinates (normalized 0.0 to 1.0) that tightly bounds ONLY the main building floor plan structure. Exclude all exterior grid lines, dimension tracks, and schedules."},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64_data}}
            ]
        }],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    
    print(f"Gemini Request: Sending image ({img_resized.size[0]}x{img_resized.size[1]}) to Gemini...")
    
    try:
        import httpx
        import asyncio
        # Use a longer timeout (60s) for complex vision tasks
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                try:
                    text = data['candidates'][0]['content']['parts'][0]['text']
                    # Handle potential markdown formatting in JSON response
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0].strip()
                    elif "```" in text:
                        text = text.split("```")[1].split("```")[0].strip()
                    
                    import json
                    parsed = json.loads(text)
                    bbox = parsed.get("building_bbox")
                    if bbox:
                        print(f"Gemini Mask Success: {bbox}")
                    return bbox
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    print(f"Gemini Parse Error: {e}. Raw text: {text if 'text' in locals() else 'N/A'}")
            else:
                print(f"Gemini API Warn: {resp.status_code} - {resp.text}")
    except asyncio.CancelledError:
        print("Gemini Mask Request was cancelled (likely client disconnect).")
        # Re-raise CancelledError to let FastAPI handle it properly, 
        # but now we have a log of where it happened.
        raise
    except Exception as e:
        print(f"Gemini Mask API Error: {type(e).__name__}: {e}")
        
    return None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "BidReady AI Model API",
        "description": "Floor Plan Object Detection using YOLOv8",
        "endpoints": {
            "/detect": "POST - Upload image for object detection",
            "/health": "GET - Health check",
            "/labels": "GET - Get available detection labels",
            "/docs": "GET - Interactive API documentation (Swagger UI)",
            "/documentation": "GET - Complete API documentation page",
            "/test": "GET - Interactive test interface",
        },
        "links": {
            "interactive_docs": "/docs",
            "full_documentation": "/documentation",
            "test_interface": "/test",
        },
    }


@app.get("/documentation", response_class=HTMLResponse)
async def get_documentation():
    """Serve the complete API documentation page"""
    try:
        with open("docs.html", "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Documentation page not found")


@app.get("/test", response_class=HTMLResponse)
async def get_test_page():
    """Serve the interactive test page"""
    try:
        with open("test.html", "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Test page not found")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        model = load_model()
        return {"status": "healthy", "model_loaded": model is not None}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/labels")
async def get_available_labels():
    """Get available detection labels"""
    return {
        "available_labels": [
            "Column",
            "Curtain Wall",
            "Dimension",
            "Door",
            "Railing",
            "Sliding Door",
            "Stair Case",
            "Wall",
            "Window",
        ]
    }


import base64
import io
from typing import Optional

import httpx
import numpy as np
import PIL.Image
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

def calculate_core_building_bbox(all_detections, img_w, img_h):
    """
    Computes a tight bounding box encapsulating the core architectural elements 
    (Doors, Windows, Stairs, Columns), effectively masking out outer grid dimension lines.
    """
    core_labels = {"Door", "Window", "Stair Case", "Column"}
    
    xmin, ymin = img_w, img_h
    xmax, ymax = 0.0, 0.0
    
    found_core = False
    
    for det in all_detections:
        label = det.get("label", "")
        # Box structure could come from `box.xyxy` in iteration or dict
        b = det if "x1" in det else det.get("bbox", {})
        if label in core_labels and "x1" in b:
            xmin = min(xmin, float(b["x1"]))
            ymin = min(ymin, float(b["y1"]))
            xmax = max(xmax, float(b["x2"]))
            ymax = max(ymax, float(b["y2"]))
            found_core = True

    if not found_core:
        return None  # No core elements found to securely mask around
        
    # Refined: Add a smaller 2% padding and ensure we don't go out of bounds
    pad_x = (xmax - xmin) * 0.02
    pad_y = (ymax - ymin) * 0.02
    
    bbox = {
        "xmin": float(max(0.0, xmin - pad_x)),
        "ymin": float(max(0.0, ymin - pad_y)),
        "xmax": float(min(img_w, xmax + pad_x)),
        "ymax": float(min(img_h, ymax + pad_y))
    }
    print(f"Fallback Mask (Core-based): {bbox}")
    return bbox

router = APIRouter()


class DetectRequest(BaseModel):
    image_url: str
    confidence: Optional[float] = 0
    selected_labels: Optional[str] = None
    use_tiling: Optional[bool] = None
    per_class_conf: Optional[Dict[str, float]] = None
    calibration: Optional[float] = None


@app.post("/detect")
async def detect_objects(req: DetectRequest):
    try:
        # Endpoint expects application/json with DetectRequest
        if not req or not getattr(req, "image_url", None):
            raise HTTPException(
                status_code=400, detail="image_url is required in JSON body"
            )

        # 1️⃣ Download Image or read local path
        try:
            if isinstance(req.image_url, str) and req.image_url.lower().startswith(
                ("http://", "https://")
            ):
                async with httpx.AsyncClient(timeout=20) as client:
                    resp = await client.get(req.image_url)
                if resp.status_code != 200:
                    raise HTTPException(
                        status_code=400, detail="Unable to download image."
                    )
                image_bytes = resp.content
            else:
                # assume it's a local filesystem path
                if not os.path.exists(req.image_url):
                    raise HTTPException(
                        status_code=400,
                        detail="Provided local image path does not exist",
                    )
                with open(req.image_url, "rb") as f:
                    image_bytes = f.read()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error fetching image: {str(e)}"
            )

        # 2️⃣ Read Image
        try:
            image = PIL.Image.open(io.BytesIO(image_bytes))
        except Exception:
            raise HTTPException(
                status_code=400, detail="Provided URL is not a valid image"
            )

        original_size = image.size
        img_w, img_h = original_size
        max_dim = max(original_size)

        # Tune tiling and model input size for higher-resolution images.
        if max_dim >= 2400:
            tile_size = 1600
            tile_overlap = tile_size // 2
            model_imgsz = 1600
        else:
            tile_size = 1200
            tile_overlap = tile_size // 2
            model_imgsz = 1280
        
        # Gemini Hybrid Validation: Grabs the tightest main architectural bounds to block exterior garbage
        gemini_bbox = await get_building_mask_from_gemini(image)
        
        padded_bbox = None
        if gemini_bbox:
            try:
                # 1. Flatten the input if it's a list containing a single string like ['381, 85, 696, 811']
                if isinstance(gemini_bbox, list) and len(gemini_bbox) == 1 and isinstance(gemini_bbox[0], str):
                    raw_coords = [c.strip() for c in gemini_bbox[0].split(",")]
                elif isinstance(gemini_bbox, str):
                    raw_coords = [c.strip() for c in gemini_bbox.split(",")]
                elif isinstance(gemini_bbox, list):
                    raw_coords = gemini_bbox
                else:
                    raw_coords = []

                if len(raw_coords) == 4:
                    ymin, xmin, ymax, xmax = map(float, raw_coords)
                
                # HEURISTIC: Gemini 1.5 often returns coordinates in 0-1000 scale 
                # even if asked for 0.0-1.0. If any value is > 1.1, assume 0-1000.
                if ymin > 1.1 or xmin > 1.1 or ymax > 1.1 or xmax > 1.1:
                    ymin /= 1000.0
                    xmin /= 1000.0
                    ymax /= 1000.0
                    xmax /= 1000.0
                
                # Allow more padding so perimeter walls/windows touching the edge don't get clipped
                pad_y = (ymax - ymin) * 0.08
                pad_x = (xmax - xmin) * 0.08
                padded_bbox = {
                    "ymin": float(max(0.0, ymin - pad_y) * img_h),
                    "xmin": float(max(0.0, xmin - pad_x) * img_w),
                    "ymax": float(min(1.0, ymax + pad_y) * img_h),
                    "xmax": float(min(1.0, xmax + pad_x) * img_w)
                }
                bbox_area = max(0.0, padded_bbox["xmax"] - padded_bbox["xmin"]) * max(
                    0.0, padded_bbox["ymax"] - padded_bbox["ymin"]
                )
                if bbox_area < (img_w * img_h * 0.35):
                    print("Gemini bbox too small, ignoring.")
                    padded_bbox = None
                else:
                    print(f"Gemini Final Boundary: {padded_bbox}")
            except (ValueError, TypeError) as e:
                print(f"Error processing Gemini bbox: {e}")
                padded_bbox = None

        # 3️⃣ Load model
        model = load_model()

        # 4️⃣ Parse labels
        available_labels = [
            "Column",
            "Curtain Wall",
            "Dimension",
            "Door",
            "Railing",
            "Sliding Door",
            "Stair Case",
            "Wall",
            "Window",
        ]

        if req.selected_labels:
            selected_labels_list = [
                label.strip() for label in req.selected_labels.split(",")
            ]
            invalid = [x for x in selected_labels_list if x not in available_labels]
            if invalid:
                raise HTTPException(
                    400, f"Invalid labels: {invalid}. Available: {available_labels}"
                )
        else:
            selected_labels_list = available_labels

        # Internal: Ensure 'Dimension' is always tracked for calibration, even if user didn't select it
        internal_labels_to_track = set(selected_labels_list)
        internal_labels_to_track.add("Dimension")
        internal_labels_to_track = list(internal_labels_to_track)

        # 5️⃣ Validate confidence
        if not 0.0 <= req.confidence <= 1.0:
            raise HTTPException(400, "Confidence must be between 0.0 and 1.0")

        # 6️⃣ Decide tiling
        use_tiling = (
            should_use_tiling(image) if req.use_tiling is None else req.use_tiling
        )

        detections = []
        all_detections = []

        # Build per-class confidence mapping (validate values and labels)
        per_class_conf_map: Dict[str, float] = {}
        if req.per_class_conf:
            for k, v in req.per_class_conf.items():
                try:
                    f = float(v)
                except Exception:
                    continue
                if 0.0 <= f <= 1.0:
                    per_class_conf_map[k] = f

        if use_tiling:
            tiles = create_tiles(image, tile_size, tile_overlap)

            for tile_img, (x_off, y_off) in tiles:
                results = model.predict(
                    tile_img, conf=req.confidence, imgsz=model_imgsz, verbose=False
                )

                for box in results[0].boxes:
                    label = model.names[int(box.cls)]
                    if label in internal_labels_to_track:
                        # Use per-class threshold if provided else default to req.confidence
                        label_min_conf = per_class_conf_map.get(label, req.confidence)
                        
                        # Aggressive Wall Boost: Drop threshold even further (0.05) to catch missing walls
                        if label == "Wall":
                            label_min_conf = min(0.03, label_min_conf)
                            
                        if float(box.conf) < label_min_conf:
                            continue
                        all_detections.append(
                            {
                                "label": label,
                                "confidence": float(box.conf),
                                "bbox": {
                                    "x1": float(box.xyxy[0][0]) + x_off,
                                    "y1": float(box.xyxy[0][1]) + y_off,
                                    "x2": float(box.xyxy[0][2]) + x_off,
                                    "y2": float(box.xyxy[0][3]) + y_off,
                                },
                            }
                        )

            detections = merge_detections(
                all_detections, original_size, iou_threshold=0.3
            )
            
            # Pure Python fallback if Gemini API fails to provide structural envelope
            if not padded_bbox:
                padded_bbox = calculate_core_building_bbox(all_detections, original_size[0], original_size[1])

            def is_valid_det(d):
                if d["label"] == "Wall" and is_false_positive_wall(d["bbox"], original_size[0], original_size[1]):
                    return False
                if padded_bbox:
                    cx = (d["bbox"]["x1"] + d["bbox"]["x2"]) / 2.0
                    cy = (d["bbox"]["y1"] + d["bbox"]["y2"]) / 2.0
                    # Check if center of detection is within the (padded) building boundary
                    if not (padded_bbox["xmin"] <= cx <= padded_bbox["xmax"] and 
                            padded_bbox["ymin"] <= cy <= padded_bbox["ymax"]):
                        return False
                return True

            detections = [d for d in detections if is_valid_det(d)]

            annotated = np.array(image)
            draw_img = PIL.Image.fromarray(annotated)
            draw = ImageDraw.Draw(draw_img)

            colors = {
                "Column": "#FF0000",
                "Curtain Wall": "#00FF00",
                "Dimension": "#0000FF",
                "Door": "#FFFF00",
                "Railing": "#FF00FF",
                "Sliding Door": "#00FFFF",
                "Stair Case": "#FFA500",
                "Wall": "#800080",
                "Window": "#FFC0CB",
            }

            for det in detections:
                b, label = det["bbox"], det["label"]
                color = colors.get(label, "#FFFFFF")
                draw.rectangle(
                    [(b["x1"], b["y1"]), (b["x2"], b["y2"])], outline=color, width=3
                )
                draw.text(
                    (b["x1"], b["y1"] - 10),
                    f"{label} {det['confidence']:.2f}",
                    fill=color,
                )

            annotated = np.array(draw_img)

        else:
            results = model.predict(image, conf=req.confidence, imgsz=model_imgsz, verbose=False)
            all_raw_detections = []
            for box in results[0].boxes:
                label = model.names[int(box.cls)]
                if label not in internal_labels_to_track:
                    continue
                
                label_min_conf = per_class_conf_map.get(label, req.confidence)
                if label == "Wall":
                    label_min_conf = min(0.05, label_min_conf)
                    
                if float(box.conf) < label_min_conf:
                    continue

                all_raw_detections.append({
                    "label": label,
                    "confidence": float(box.conf),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3]),
                    },
                })

            # Pure Python fallback if Gemini API fails to provide structural envelope
            if not padded_bbox:
                padded_bbox = calculate_core_building_bbox(all_raw_detections, original_size[0], original_size[1])

            def is_valid_det(d):
                if d["label"] == "Wall" and is_false_positive_wall(d["bbox"], original_size[0], original_size[1]):
                    return False
                if padded_bbox:
                    cx = (d["bbox"]["x1"] + d["bbox"]["x2"]) / 2.0
                    cy = (d["bbox"]["y1"] + d["bbox"]["y2"]) / 2.0
                    if not (padded_bbox["xmin"] <= cx <= padded_bbox["xmax"] and 
                            padded_bbox["ymin"] <= cy <= padded_bbox["ymax"]):
                        return False
                return True

            detections = [d for d in all_raw_detections if is_valid_det(d)]

        # --- UNIFIED ANNOTATION LOGIC ---
        annotated = np.array(image.convert("RGB"))
        draw_img = PIL.Image.fromarray(annotated)
        draw = ImageDraw.Draw(draw_img)

        colors = {
            "Column": "#FF0000",
            "Curtain Wall": "#00FF00",
            "Dimension": "#0000FF",
            "Door": "#FFFF00",
            "Railing": "#FF00FF",
            "Sliding Door": "#00FFFF",
            "Stair Case": "#FFA500",
            "Wall": "#800080",
            "Window": "#FFC0CB",
        }

        for det in detections:
            b, label = det["bbox"], det["label"]
            color = colors.get(label, "#FFFFFF")
            # Draw rectangle
            draw.rectangle(
                [(b["x1"], b["y1"]), (b["x2"], b["y2"])], 
                outline=color, 
                width=max(2, int(min(original_size) / 500))
            )
            # Draw label
            draw.text(
                (b["x1"], b["y1"] - 15),
                f"{label} {det['confidence']:.2f}",
                fill=color,
            )

        # Optional: Draw the building mask (Gemini boundary) in blue for visual confirmation
        if padded_bbox:
            draw.rectangle(
                [(padded_bbox["xmin"], padded_bbox["ymin"]), (padded_bbox["xmax"], padded_bbox["ymax"])],
                outline="#0000FF",
                width=5
            )

        annotated = np.array(draw_img)

        # Count objects
        counts = {}
        for d in detections:
            counts[d["label"]] = counts.get(d["label"], 0) + 1

        # Base64 encode image
        buf = io.BytesIO()
        PIL.Image.fromarray(annotated).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        # 7️⃣ Attempt to compute px_per_inch from Dimension detections or request
        px_per_inch = (
            req.calibration if req.calibration and req.calibration > 0 else None
        )
        dimension_info = None

        # Also surface up to 5 smallest Dimension boxes for external OCR use
        dimension_candidates: list = []

        # Find ALL Dimension detections
        dimension_detections = [d for d in detections if d["label"] == "Dimension"]

        if dimension_detections:
            # Compute area for each detection and keep a compact copy for response
            def _area(det):
                b = det["bbox"]
                return max(0.0, (b["x2"] - b["x1"])) * max(0.0, (b["y2"] - b["y1"]))

            # Top 5 smallest by area for client-side OCR or fallback flows
            sorted_by_area = sorted(dimension_detections, key=_area)
            for det in sorted_by_area[:5]:
                b = det["bbox"]
                dimension_candidates.append(
                    {
                        "bbox": {
                            "x1": float(b["x1"]),
                            "y1": float(b["y1"]),
                            "x2": float(b["x2"]),
                            "y2": float(b["y2"]),
                        },
                        "confidence": float(det.get("confidence", 0.0)),
                        "area_px": float(_area(det)),
                    }
                )

            # Smart selection: Try top 3 smallest bboxes (likely actual dimension text, not lines)
            def score_dimension(det):
                bbox = det["bbox"]
                width = bbox["x2"] - bbox["x1"]
                height = bbox["y2"] - bbox["y1"]
                area = width * height
                # Prefer SMALLER boxes (actual text annotations are compact)
                return -area  # Negative so smallest comes first

            # Sort by score and take top 3 smallest candidates
            dimension_detections.sort(key=score_dimension, reverse=True)

            if px_per_inch:
                dimension_info = {
                    "px_per_inch": px_per_inch,
                    "method": "manual_override",
                }
                top_candidates = []
            else:
                top_candidates = dimension_detections[:3]

            # Create debug directory if it doesn't exist
            debug_dir = "debug_dimension_crops"
            os.makedirs(debug_dir, exist_ok=True)

            attempted_results = []

            for dim_idx, dim_detection in enumerate(top_candidates):
                try:
                    # Extract text using AWS Rekognition (NO upscaling needed)
                    dim_text = extract_text_from_bbox_rekognition(
                        image_bytes, dim_detection["bbox"]
                    )

                    # Log what we extracted
                    attempt_info = {
                        "bbox_index": dim_idx,
                        "bbox": dim_detection["bbox"],
                        "ocr_text": dim_text,
                    }

                    if dim_text and len(dim_text) > 0:
                        # Try to parse and compute px_per_inch
                        try:
                            px_per_inch, px_length, real_inches = (
                                compute_px_per_inch_from_dimension(
                                    image_bytes, dim_detection["bbox"], dim_text
                                )
                            )
                            # SUCCESS!
                            dimension_info = {
                                "text": dim_text,
                                "px_length": px_length,
                                "real_inches": real_inches,
                                "px_per_inch": px_per_inch,
                                "bbox_used": dim_detection["bbox"],
                                "detection_index": dim_idx,
                                "total_detections": len(dimension_detections),
                                "method": "aws_rekognition",
                            }
                            break  # Success! Stop trying
                        except Exception as parse_error:
                            attempt_info["parse_error"] = str(parse_error)
                            attempted_results.append(attempt_info)
                            continue
                    else:
                        attempt_info["error"] = "No text extracted from bbox"
                        attempted_results.append(attempt_info)
                        continue

                except Exception as e:
                    attempted_results.append(
                        {
                            "bbox_index": dim_idx,
                            "bbox": dim_detection["bbox"],
                            "error": str(e),
                        }
                    )
                    continue

            # If we didn't find any valid dimension, report attempts
            if not dimension_info:
                dimension_info = {
                    "error": "Could not extract valid dimension from any bbox",
                    "total_dimension_detections": len(dimension_detections),
                    "candidates_tried": len(top_candidates),
                    "attempts": attempted_results,
                    "method": "aws_rekognition",
                }

        # 8️⃣ Attempt to detect and parse scale information
        scale_info = None
        scale_ratio = None

        # Look for detections that might contain scale info (could be labeled as "Dimension" or text regions)
        # OPTIMIZATION: Only look at 'Dimension' labels to avoid making 100s of OCR calls on Walls/Doors
        scale_candidates = [d for d in detections if d["label"] == "Dimension"]

        # We'll scan candidate detections and try OCR to find scale text
        for det in scale_candidates:
            # Skip if we already found a valid scale
            if scale_ratio:
                break

            # Try to extract text and check if it contains scale info (use image_bytes)
            try:
                text = extract_text_from_bbox_rekognition(image_bytes, det["bbox"])
                if text and any(
                    keyword in text.upper()
                    for keyword in ["SCALE", "=", ":", "NTS", "NOT TO SCALE"]
                ):
                    try:
                        parsed_scale = parse_scale_text(text)
                        scale_ratio = parsed_scale.get("ratio")
                        scale_info = parsed_scale
                        break
                    except Exception:
                        # Not a valid scale text, continue searching
                        continue
            except Exception:
                continue

        # Fallback: If no scale found in detections, scan the bottom 20% of the image (typical title block area)
        if not scale_info:
            try:
                # Define bbox for bottom 20%
                height = original_size[1]
                width = original_size[0]
                bottom_bbox = {
                    "x1": 0,
                    "y1": int(height * 0.8),
                    "x2": width,
                    "y2": height,
                }

                # Try OCR on the title block area
                # Note: This adds one OCR call but handles cases where YOLO missed the text bbox
                tb_text = extract_text_from_bbox_rekognition(image_bytes, bottom_bbox)
                if tb_text and any(
                    keyword in tb_text.upper()
                    for keyword in ["SCALE", "=", ":", "NTS", "NOT TO SCALE"]
                ):
                    parsed_scale = parse_scale_text(tb_text)
                    if parsed_scale.get("ratio"):
                        scale_ratio = parsed_scale.get("ratio")
                        scale_info = parsed_scale
                        # Enforce source type for debugging
                        scale_info["source"] = "title_block_fallback"
            except Exception:
                pass

        # 9️⃣ Get shapes and convert areas with scale applied (use image_bytes)
        shapes_with_colors = detect_shapes(image_bytes, colorize=True)

        # Add area measurements to each shape
        if px_per_inch:
            for shape in shapes_with_colors:
                area_px = shape.get("area", 0)
                # Drawing area in square inches (on the floor plan)
                drawing_sq_in = convert_area_px_to_sqin(area_px, px_per_inch)
                shape["area_sq_in"] = drawing_sq_in

                # Compute square feet
                # If we have scale, apply it; otherwise treat as 1:1 (drawing = reality)
                if scale_ratio and scale_ratio > 0:
                    actual_sq_ft = compute_actual_sqft_from_drawing(
                        area_px, px_per_inch, scale_ratio
                    )
                else:
                    # No scale detected, assume 1:1 (drawing measurements are real measurements)
                    actual_sq_ft = drawing_sq_in / 144.0

                shape["area_sq_ft"] = actual_sq_ft

        shapes = shapes_with_colors

        # Build an SVG overlay using the colors from detect_shapes (if present)
        try:
            svg_overlay = build_svg_from_paths(
                shapes_with_colors,
                original_size[0],
                original_size[1],
                stroke_color="#0b61e9",
                stroke_width=2,
                svg_fill="none",
                fill_opacity=0.12,
            )
        except Exception:
            svg_overlay = None

        response_data = {
            "success": True,
            "total_detections": len(detections),
            "object_counts": counts,
            "detections": detections,
            "parameters": {
                "confidence": req.confidence,
                "selected_labels": selected_labels_list,
                "tiling_used": use_tiling,
                "original_image_size": {
                    "width": original_size[0],
                    "height": original_size[1],
                },
            },
            "shapes": shapes,
            "shapes_svg": svg_overlay,
        }
        # Always include up to 5 smallest Dimension boxes for OCR on client if needed
        if dimension_candidates:
            response_data["dimension_candidates"] = dimension_candidates
        if per_class_conf_map:
            response_data["per_class_conf"] = per_class_conf_map

        # FINAL FILTER: Remove objects from 'detections' if user didn't ask for them
        # (e.g. we forced 'Dimension' for calibration, but user might not want to see it)
        final_detections = []
        for det in response_data["detections"]:
            if det["label"] in selected_labels_list:
                final_detections.append(det)
        response_data["detections"] = final_detections

        # Add dimension info if available
        if dimension_info:
            response_data["dimension_calibration"] = dimension_info

        # Add scale info if available
        if scale_info:
            response_data["scale_info"] = scale_info

        return response_data

    except Exception as e:
        raise HTTPException(500, f"Detection failed: {str(e)}")


@app.post("/detect-simple")
async def detect_objects_simple(file: UploadFile = File(...)):
    """
    Simplified detection endpoint with default parameters
    """
    # Save upload to temporary file and call the main detect endpoint logic via DetectRequest
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(file.filename)[1] if file.filename else ".png",
        )
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()
        # Build a DetectRequest targeting the local temp path (detect_objects supports local paths)
        req = DetectRequest(image_url=tmp.name, confidence=0.25)
        response = await detect_objects(req)
        return response
    finally:
        if tmp and os.path.exists(tmp.name):
            try:
                os.remove(tmp.name)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
