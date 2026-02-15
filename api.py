from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import cv2
from scripts.inference import run_inference

app = FastAPI()

UPLOAD_DIR = "temp"
OUTPUT_DIR = "data/output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def health():
    return {"status": "API running"}

# -----------------------------
# Image Prediction Endpoint
# -----------------------------
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):

    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    output_path = os.path.join(OUTPUT_DIR, f"annotated_{file.filename}")

    # Save uploaded file
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run inference
    annotated_frame, detections = run_inference(temp_path)

    # Save annotated image
    cv2.imwrite(output_path, annotated_frame)

    # Cleanup temp file
    os.remove(temp_path)

    return JSONResponse({
        "detections": detections,
        "output_image": output_path
    })
