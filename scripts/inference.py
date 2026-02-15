from ultralytics import YOLO
import cv2
import os

# -----------------------------
# Model Path
# -----------------------------
MODEL_PATH = "../models/best.pt"

# -----------------------------
# Load model ONCE
# -----------------------------
model = YOLO(MODEL_PATH)
print("✅ Model loaded")

# -----------------------------
# Inference Function
# -----------------------------
def run_inference(image_path, conf=0.25, imgsz=640):

    results = model(
        image_path,
        imgsz=imgsz,
        conf=conf
    )

    annotated_frame = results[0].plot()

    detections = {
        "boxes": results[0].boxes.xyxy.tolist() if results[0].boxes else [],
        "scores": results[0].boxes.conf.tolist() if results[0].boxes else [],
        "classes": results[0].boxes.cls.tolist() if results[0].boxes else []
    }

    return annotated_frame, detections
