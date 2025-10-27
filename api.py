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
import helper
import json
import os
import cv2

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

def create_tiles(image: PIL.Image.Image, tile_size: int = 1280, overlap: int = 200) -> List[Tuple[PIL.Image.Image, Tuple[int, int]]]:
    """
    Split large image into overlapping tiles for better detection
    
    Args:
        image: Input PIL Image
        tile_size: Size of each tile (default 1280x1280)
        overlap: Overlap between tiles in pixels (default 200)
    
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
    
    # Convert to numpy arrays for NMS
    boxes = []
    scores = []
    labels = []
    
    for det in all_detections:
        bbox = det['bbox']
        boxes.append([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
        scores.append(det['confidence'])
        labels.append(det['label'])
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Apply NMS per class
    keep_indices = []
    unique_labels = set(labels)
    
    for label in unique_labels:
        label_mask = np.array([l == label for l in labels])
        label_boxes = boxes[label_mask]
        label_scores = scores[label_mask]
        label_indices = np.where(label_mask)[0]
        
        if len(label_boxes) > 0:
            # Apply NMS using cv2
            indices = cv2.dnn.NMSBoxes(
                label_boxes.tolist(),
                label_scores.tolist(),
                score_threshold=0.0,
                nms_threshold=iou_threshold
            )
            
            if len(indices) > 0:
                keep_indices.extend(label_indices[indices.flatten()].tolist())
    
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

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    confidence: Optional[float] = 0.25,
    selected_labels: Optional[str] = None,
    use_tiling: Optional[bool] = None
):
    """
    Detect objects in uploaded image
    
    Args:
        file: Image file (jpg, jpeg, png)
        confidence: Detection confidence threshold (0.0 to 1.0)
        selected_labels: Comma-separated list of labels to detect (optional)
        use_tiling: Force tiling on/off (auto-detect if None)
    
    Returns:
        JSON with detection results, counts, and annotated image
    """
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only JPG, JPEG, and PNG are supported."
            )
        
        # Read and process image
        contents = await file.read()
        image = PIL.Image.open(io.BytesIO(contents))
        original_size = image.size
        
        # Load model
        model = load_model()
        
        # Parse selected labels
        available_labels = ['Column', 'Curtain Wall', 'Dimension', 'Door', 'Railing', 'Sliding Door', 'Stair Case', 'Wall', 'Window']
        if selected_labels:
            selected_labels_list = [label.strip() for label in selected_labels.split(',')]
            # Validate labels
            invalid_labels = [label for label in selected_labels_list if label not in available_labels]
            if invalid_labels:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid labels: {invalid_labels}. Available labels: {available_labels}"
                )
        else:
            selected_labels_list = available_labels
        
        # Validate confidence
        if not 0.0 <= confidence <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="Confidence must be between 0.0 and 1.0"
            )
        
        # Determine if we should use tiling
        if use_tiling is None:
            use_tiling = should_use_tiling(image)
        
        all_detections = []
        
        if use_tiling:
            # Process image with tiling for better detection on large images
            tiles = create_tiles(image, tile_size=1280, overlap=200)
            
            for tile_img, (x_offset, y_offset) in tiles:
                # Run prediction on tile
                results = model.predict(tile_img, conf=confidence, verbose=False)
                
                # Process detections from this tile
                for box in results[0].boxes:
                    label = model.names[int(box.cls)]
                    if label in selected_labels_list:
                        # Adjust coordinates to original image space
                        detection = {
                            "label": label,
                            "confidence": float(box.conf),
                            "bbox": {
                                "x1": float(box.xyxy[0][0]) + x_offset,
                                "y1": float(box.xyxy[0][1]) + y_offset,
                                "x2": float(box.xyxy[0][2]) + x_offset,
                                "y2": float(box.xyxy[0][3]) + y_offset
                            }
                        }
                        all_detections.append(detection)
            
            # Merge overlapping detections
            detections = merge_detections(all_detections, original_size, iou_threshold=0.5)
            
            # Create annotated image manually
            annotated_image = np.array(image)
            annotated_pil = PIL.Image.fromarray(annotated_image)
            draw = ImageDraw.Draw(annotated_pil)
            
            # Define colors for different labels
            colors = {
                'Column': '#FF0000', 'Curtain Wall': '#00FF00', 'Dimension': '#0000FF',
                'Door': '#FFFF00', 'Railing': '#FF00FF', 'Sliding Door': '#00FFFF',
                'Stair Case': '#FFA500', 'Wall': '#800080', 'Window': '#FFC0CB'
            }
            
            for det in detections:
                bbox = det['bbox']
                label = det['label']
                color = colors.get(label, '#FFFFFF')
                
                # Draw bounding box
                draw.rectangle(
                    [(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2'])],
                    outline=color,
                    width=3
                )
                
                # Draw label
                text = f"{label} {det['confidence']:.2f}"
                draw.text((bbox['x1'], bbox['y1'] - 10), text, fill=color)
            
            annotated_image = np.array(annotated_pil)
            
        else:
            # Process normally for smaller images
            results = model.predict(image, conf=confidence, imgsz=1280)
            
            # Filter results by selected labels
            filtered_boxes = [
                box for box in results[0].boxes 
                if model.names[int(box.cls)] in selected_labels_list
            ]
            
            # Prepare detection details
            detections = []
            for box in filtered_boxes:
                detection = {
                    "label": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    }
                }
                detections.append(detection)
            
            # Generate annotated image
            results[0].boxes = filtered_boxes
            annotated_image = results[0].plot()[:, :, ::-1]  # Convert BGR to RGB
        
        # Count detected objects
        object_counts = {}
        for det in detections:
            label = det['label']
            object_counts[label] = object_counts.get(label, 0) + 1
        
        # Convert annotated image to base64
        pil_image = PIL.Image.fromarray(annotated_image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        annotated_image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "success": True,
            "total_detections": len(detections),
            "object_counts": object_counts,
            "detections": detections,
            "annotated_image": f"data:image/png;base64,{annotated_image_b64}",
            "parameters": {
                "confidence": confidence,
                "selected_labels": selected_labels_list,
                "tiling_used": use_tiling,
                "original_image_size": {"width": original_size[0], "height": original_size[1]}
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect-simple")
async def detect_objects_simple(file: UploadFile = File(...)):
    """
    Simplified detection endpoint with default parameters
    """
    return await detect_objects(file=file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)