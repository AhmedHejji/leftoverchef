from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import json

app = FastAPI()

# Load model once at startup (CPU mode by default)
model = YOLO('yolov8n.pt')

@app.get("/healthcheck")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        results = model(img)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="Inference failed")

    # Format results, adding class_name lookup
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy().flatten()
    cls_ids = results[0].boxes.cls.cpu().numpy().astype(int).flatten()

    preds = []
    for (x1, y1, x2, y2), conf, cid in zip(boxes, confs, cls_ids):
        name = model.names[cid] if cid in model.names else str(cid)
        preds.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": float(conf),
            "class_id": int(cid),
            "class_name": name
        })

    return JSONResponse(content={"predictions": preds})
