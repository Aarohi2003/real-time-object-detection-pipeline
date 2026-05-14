# Real-Time Object Detection Pipeline

## Overview
This project demonstrates an end-to-end Object Detection Pipeline using concepts from RCNN and modern YOLOv8 architecture.

The pipeline explores:
- Region Proposal Generation using Selective Search
- CNN Feature Extraction using ResNet50
- Feature Map Visualization
- YOLOv8 Real-Time Detection
- Webcam-based Object Detection

---

## Features
✅ Selective Search Region Proposals  
✅ CNN Feature Extraction (ResNet50)  
✅ Real-Time Object Detection  
✅ YOLOv8 Inference  
✅ Feature Map Visualization  
✅ OpenCV Integration  
✅ Deep Learning Pipeline Workflow  

---

## Tech Stack
- Python
- OpenCV
- PyTorch
- TensorFlow / Keras
- YOLOv8
- ResNet50
- Matplotlib

---

## Pipeline Workflow

Input Image  
→ Selective Search  
→ Region Proposals  
→ Crop & Resize  
→ CNN Feature Extraction  
→ Feature Maps  
→ Detection & Classification  

---

## Project Structure

```bash
├── object_detection_pipeline.ipynb
├── images/
├── outputs/
├── README.md
```

---

## Installation

```bash
pip install ultralytics
pip install opencv-contrib-python
pip install tensorflow
pip install torch torchvision
```

---

## Run YOLOv8 Detection

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("image.jpg")
```

---

## Real-Time Webcam Detection

```python
cap = cv2.VideoCapture(0)
```

---

## Future Improvements
- Faster RCNN implementation
- Custom dataset training
- Object tracking integration
- Deployment using Streamlit

---

## Author
Aarohi Sharma  
AI & Computer Vision Enthusiast
