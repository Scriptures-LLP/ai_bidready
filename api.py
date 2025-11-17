from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import PIL.Image
from PIL import ImageDraw
import torch
import io
import base64
import numpy as np
from typing import List, Optional, Tuple
import os
import tempfile
import cv2
from service.detect import detect_shapes, build_svg_from_paths, extract_text_from_bbox_ocr, compute_px_per_inch_from_dimension, convert_area_px_to_sqin

app = FastAPI(
    title="BidReady AI Model API",
    description="Floor Plan Object Detection API using YOLOv8",
    version="1.0.0"
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

def load_model():
    """Load the YOLO model with proper torch configuration"""
    global model
    if model is None:
        # Temporarily patch torch.load to use weights_only=False
        original_torch_load = torch.load
        def patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        torch.load = patched_torch_load
        try:
            model = YOLO('best.pt')
        finally:
            torch.load = original_torch_load
    return model

def create_tiles(image: PIL.Image.Image, tile_size: int = 1920, overlap: int = 320) -> List[Tuple[PIL.Image.Image, Tuple[int, int]]]:
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

def merge_detections(all_detections: List, image_size: Tuple[int, int], iou_threshold: float = 0.5):
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
        bbox = det['bbox']
        boxes_xyxy.append([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
        scores.append(det['confidence'])
        labels.append(det['label'])

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
                nms_threshold=float(iou_threshold)
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
            "/test": "GET - Interactive test interface"
        },
        "links": {
            "interactive_docs": "/docs",
            "full_documentation": "/documentation",
            "test_interface": "/test"
        }
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
            'Column', 'Curtain Wall', 'Dimension', 'Door', 
            'Railing', 'Sliding Door', 'Stair Case', 'Wall', 'Window'
        ]
    }

from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from pydantic import BaseModel
import httpx
import io
import PIL.Image
import base64
import numpy as np
from typing import Optional

router = APIRouter()

class DetectRequest(BaseModel):
    image_url: str
    confidence: Optional[float] = 0.25
    selected_labels: Optional[str] = None
    use_tiling: Optional[bool] = None

@app.post("/detect")
async def detect_objects(
    req: Optional[DetectRequest] = None,
    image_url: Optional[str] = Form(None),
    confidence: Optional[float] = Form(None),
    selected_labels: Optional[str] = Form(None),
    use_tiling: Optional[str] = Form(None),
):
    try:
        # If the client posted form-data instead of JSON body, fill `req` from the form fields.
        if req is None:
            # require image_url from form
            if not image_url:
                raise HTTPException(status_code=400, detail="image_url is required")

            # parse optional values; confidence expects float, use_tiling can be truthy string
            parsed_confidence = float(confidence) if confidence is not None else 0.25
            parsed_use_tiling = None
            if use_tiling is not None:
                # convert common string forms to bool
                val = str(use_tiling).strip().lower()
                if val in ("1", "true", "t", "yes", "y", "on"):
                    parsed_use_tiling = True
                elif val in ("0", "false", "f", "no", "n", "off"):
                    parsed_use_tiling = False
                else:
                    parsed_use_tiling = None

            req = DetectRequest(
                image_url=image_url,
                confidence=parsed_confidence,
                selected_labels=selected_labels,
                use_tiling=parsed_use_tiling,
            )

        # 1️⃣ Download Image or read local path
        try:
            if isinstance(req.image_url, str) and req.image_url.lower().startswith(("http://", "https://")):
                async with httpx.AsyncClient(timeout=20) as client:
                    resp = await client.get(req.image_url)
                if resp.status_code != 200:
                    raise HTTPException(status_code=400, detail="Unable to download image.")
                image_bytes = resp.content
            else:
                # assume it's a local filesystem path
                if not os.path.exists(req.image_url):
                    raise HTTPException(status_code=400, detail="Provided local image path does not exist")
                with open(req.image_url, 'rb') as f:
                    image_bytes = f.read()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")

        # 2️⃣ Read Image
        try:
            image = PIL.Image.open(io.BytesIO(image_bytes))
        except Exception:
            raise HTTPException(status_code=400, detail="Provided URL is not a valid image")

        original_size = image.size
        
        # 3️⃣ Load model
        model = load_model()

        # 4️⃣ Parse labels
        available_labels = [
            'Column', 'Curtain Wall', 'Dimension', 'Door', 'Railing', 
            'Sliding Door', 'Stair Case', 'Wall', 'Window'
        ]

        if req.selected_labels:
            selected_labels_list = [label.strip() for label in req.selected_labels.split(',')]
            invalid = [x for x in selected_labels_list if x not in available_labels]
            if invalid:
                raise HTTPException(
                    400, f"Invalid labels: {invalid}. Available: {available_labels}"
                )
        else:
            selected_labels_list = available_labels

        # 5️⃣ Validate confidence
        if not 0.0 <= req.confidence <= 1.0:
            raise HTTPException(400, "Confidence must be between 0.0 and 1.0")

        # 6️⃣ Decide tiling
        use_tiling = should_use_tiling(image) if req.use_tiling is None else req.use_tiling

        detections = []
        all_detections = []

        if use_tiling:
            tiles = create_tiles(image, 1920, 920)
            
            for tile_img, (x_off, y_off) in tiles:
                results = model.predict(tile_img, conf=req.confidence, imgsz=1280, verbose=False)
                
                for box in results[0].boxes:
                    label = model.names[int(box.cls)]
                    if label in selected_labels_list:
                        all_detections.append({
                            "label": label,
                            "confidence": float(box.conf),
                            "bbox": {
                                "x1": float(box.xyxy[0][0]) + x_off,
                                "y1": float(box.xyxy[0][1]) + y_off,
                                "x2": float(box.xyxy[0][2]) + x_off,
                                "y2": float(box.xyxy[0][3]) + y_off
                            }
                        })

            detections = merge_detections(all_detections, original_size, 0.3)

            annotated = np.array(image)
            draw_img = PIL.Image.fromarray(annotated)
            draw = ImageDraw.Draw(draw_img)

            colors = {
                'Column': '#FF0000', 'Curtain Wall': '#00FF00', 'Dimension': '#0000FF',
                'Door': '#FFFF00', 'Railing': '#FF00FF', 'Sliding Door': '#00FFFF',
                'Stair Case': '#FFA500', 'Wall': '#800080', 'Window': '#FFC0CB'
            }

            for det in detections:
                b, label = det['bbox'], det['label']
                color = colors.get(label, '#FFFFFF')
                draw.rectangle([(b['x1'], b['y1']), (b['x2'], b['y2'])], outline=color, width=3)
                draw.text((b['x1'], b['y1'] - 10), f"{label} {det['confidence']:.2f}", fill=color)

            annotated = np.array(draw_img)

        else:
            results = model.predict(image, conf=req.confidence, imgsz=1280)
            filtered = [box for box in results[0].boxes if model.names[int(box.cls)] in selected_labels_list]

            for box in filtered:
                detections.append({
                    "label": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    }
                })

            results[0].boxes = filtered
            annotated = results[0].plot()[:, :, ::-1]

        # Count objects
        counts = {}
        for d in detections:
            counts[d['label']] = counts.get(d['label'], 0) + 1

        # Base64 encode image
        buf = io.BytesIO()
        PIL.Image.fromarray(annotated).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        
        # 7️⃣ Attempt to compute px_per_inch from first Dimension detection
        px_per_inch = None
        dimension_info = None
        
        # Find first Dimension detection
        dimension_detections = [d for d in detections if d['label'] == 'Dimension']
        if dimension_detections:
            first_dim = dimension_detections[0]
            try:
                # Extract text from dimension bbox using OCR
                dim_text = extract_text_from_bbox_ocr(req.image_url, first_dim['bbox'])
                if dim_text:
                    # Compute px_per_inch
                    px_per_inch, px_length, real_inches = compute_px_per_inch_from_dimension(
                        req.image_url, first_dim['bbox'], dim_text
                    )
                    dimension_info = {
                        "text": dim_text,
                        "px_length": px_length,
                        "real_inches": real_inches,
                        "px_per_inch": px_per_inch
                    }
            except Exception as e:
                # If dimension parsing fails, continue without conversion
                dimension_info = {"error": str(e)}
        
        # 8️⃣ Get shapes and convert areas to square inches if px_per_inch available
        shapes_with_colors = detect_shapes(req.image_url, colorize=True)
        
        # Add area_sq_in to each shape if px_per_inch computed
        if px_per_inch:
            for shape in shapes_with_colors:
                area_px = shape.get('area', 0)
                shape['area_sq_in'] = convert_area_px_to_sqin(area_px, px_per_inch)
        
        shapes = shapes_with_colors

        # Build an SVG overlay using the colors from detect_shapes (if present)
        try:
            svg_overlay = build_svg_from_paths(shapes_with_colors, original_size[0], original_size[1], stroke_color='#0b61e9', stroke_width=2, svg_fill='none', fill_opacity=0.12)
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
                "original_image_size": {"width": original_size[0], "height": original_size[1]}
            },
            "shapes": shapes,
            "shapes_svg": svg_overlay,
        }
        
        # Add dimension info if available
        if dimension_info:
            response_data["dimension_calibration"] = dimension_info
        
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
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] if file.filename else '.png')
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
