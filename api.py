from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import PIL.Image
import torch
import io
import base64
import numpy as np
from typing import List, Optional
import helper
import json

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
            "/labels": "GET - Get available detection labels"
        }
    }

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
    selected_labels: Optional[str] = None
):
    """
    Detect objects in uploaded image
    
    Args:
        file: Image file (jpg, jpeg, png)
        confidence: Detection confidence threshold (0.0 to 1.0)
        selected_labels: Comma-separated list of labels to detect (optional)
    
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
        
        # Run prediction
        results = model.predict(image, conf=confidence)
        
        # Filter results by selected labels
        filtered_boxes = [
            box for box in results[0].boxes 
            if model.names[int(box.cls)] in selected_labels_list
        ]
        
        # Count detected objects
        object_counts = helper.count_detected_objects(model, filtered_boxes)
        
        # Generate annotated image
        results[0].boxes = filtered_boxes
        annotated_image = results[0].plot()[:, :, ::-1]  # Convert BGR to RGB
        
        # Convert annotated image to base64
        pil_image = PIL.Image.fromarray(annotated_image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        annotated_image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
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
        
        return {
            "success": True,
            "total_detections": len(filtered_boxes),
            "object_counts": object_counts,
            "detections": detections,
            "annotated_image": f"data:image/png;base64,{annotated_image_b64}",
            "parameters": {
                "confidence": confidence,
                "selected_labels": selected_labels_list
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