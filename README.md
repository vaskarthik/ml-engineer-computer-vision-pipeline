# 🚀 ML Engineer – Computer Vision Pipeline  
### End-to-End Object Detection System (COCO → YOLO → FastAPI)

---

## 📌 Overview

This project demonstrates a complete **machine learning engineering pipeline** for **object detection**, starting from raw dataset preparation to a deployed inference API.

It highlights practical skills in:

✔ Dataset preprocessing  
✔ Annotation format conversion  
✔ Deep learning model training  
✔ Inference validation  
✔ API-based ML deployment  

---

## 🎯 Project Objective

To build a real-world **object detection pipeline**:
Raw Dataset → Annotation Conversion → Model Training → Inference → Deployment



## 🧭 Development Workflow

---

### **1️⃣ Dataset Selection**
- Dataset used: **COCO (Common Objects in Context)**
- Rich multi-class object annotations
- JSON-based bounding box format

---

### **2️⃣ Annotation Conversion**
Converted dataset annotations: COCO JSON → YOLO TXT


### **3️⃣ Model Training**
Training performed on:

🚀 **Google Colab (GPU Enabled)**

Steps:

✔ Uploaded processed dataset  
✔ Configured YOLO training  
✔ Trained custom detector  
✔ Obtained trained weights  

Output: best.pt

### **4️⃣ Model Export**
Downloaded trained weights and stored locally: models/best.pt

### **5️⃣ Local Inference**
Validated:

✔ Model loading  
✔ Detection accuracy  
✔ Bounding box visualization  

Generated:

✔ Annotated detection images  

---

### **6️⃣ FastAPI Deployment**
Wrapped inference pipeline using:

⚡ **FastAPI**

Features:

✔ REST API endpoint  
✔ Image upload support  
✔ JSON detection response  
✔ Annotated image saving  



---

## ⚙️ Environment Setup

### **1️⃣ Clone Repository**

```bash
git clone https://github.com/vaskarthik/ml-engineer-computer-vision-pipeline.git
cd ml-engineer-computer-vision-pipeline


2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux / Mac

3️⃣ Install Dependencies
python -m pip install -r requirements.txt

🧪 Run Local Inference
python scripts/inference.py

✔ Loads YOLO model
✔ Runs detection
✔ Saves annotated output


🌐 Run FastAPI Server
python -m uvicorn api:app --reload


Server URL: http://127.0.0.1:8000


Swagger UI: http://127.0.0.1:8000/docs



📷 API Endpoint
POST /predict/image

Upload image → Receive:

✔ Bounding boxes
✔ Confidence scores
✔ Class predictions
✔ Annotated output


🛠 Technologies Used

Python

Ultralytics YOLO

OpenCV

FastAPI

Uvicorn

Google Colab (GPU Training)



💡 Key Skills Demonstrated

✔ ML Pipeline Design
✔ Dataset Annotation Conversion
✔ YOLO Training
✔ Model Export / Import
✔ Inference Validation
✔ API Deployment


🚀 Future Enhancements

✔ Video inference endpoint
✔ Batch inference
✔ Docker containerization
✔ Cloud deployment
✔ Performance optimization

👨‍💻 Author

Karthik Vas S
Machine Learning / Computer Vision Engineer



⭐ Takeaway

This repository demonstrates a practical ML engineering workflow:

✅ Data → Model → Deployment