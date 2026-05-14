# RCNN

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch # Similar to tensorflow, keras -> Pytorch library, another way to implement CNNs
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights # Step 3
from torchvision import transforms
from skimage import data

!pip uninstall opencv-python opencv-contrib-python -y
!pip install opencv-contrib-python

import cv2
image = cv2.imread("/content/Kartik_sir_snapshot.png")

# Create selective search object
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)

# Selective Search Segmentation -> Quality / Fast
ss.switchToSelectiveSearchFast()
rects = ss.process()

len(rects)
filtered_proposals = []

for (x, y, w, h) in rects:
  if w > 80 and h > 80:
    filtered_proposals.append((x, y, w, h))
len(filtered_proposals)

filtered_proposals = filtered_proposals[:5]

import matplotlib.pyplot as plt
import matplotlib.patches as patches


fig, ax = plt.subplots(1, figsize=(6, 5))
ax.imshow(image)

for i, (x, y, w, h) in enumerate(proposals):
  rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
  ax.add_patch(rect)

plt.title('Region Proposals')
plt.axis('off')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

FIXED_SIZE = (224,224)
warped_regions = []

fig, axes = plt.subplots(1, len(filtered_proposals), figsize=(12, 3))

for i, (x, y, w, h) in enumerate(filtered_proposals):
    # Crop the image using numpy array slicing [y_start:y_end, x_start:x_end]
    cropped_region = image[y:y+h, x:x+w]

    # Warp (resize) the crop to the required 224x224 dimensions
    warped_region = cv2.resize(cropped_region, FIXED_SIZE)
    warped_regions.append(warped_region)

    # VISUALIZATION
    axes[i].imshow(warped_region)
    axes[i].set_title(f"Proposal {i+1}\nResized to {FIXED_SIZE}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Is this the model of step 3, or do I need to add something else
batch_images = np.array(warped_regions, dtype=np.float32)

# Apply standard ImageNet preprocessing for ResNet50 in TensorFlow
# This handles the specific mean/std normalization automatically
batch_tensors = preprocess_input(batch_images)

model = ResNet50(weights='imagenet', pooling='avg', include_top=False) # ****


features = model.predict(batch_tensors)

# VISUALIZATION: Plot the actual feature activations
plt.figure(figsize=(10, 3))
# Displaying the first 50 features of the 2048 for visual clarity
# Since 'features' is already a numpy array in TF, we don't need .numpy() like in PyTorch
plt.imshow(features[:, :50], aspect='auto', cmap='plasma')
plt.title("Step 3: Actual CNN Feature Maps (First 50 dimensions of 2048)")
plt.xlabel("Feature Dimensions")
plt.ylabel("Proposal Index")
plt.colorbar(label='Activation Magnitude')
plt.show()


# yolo model

# Step 1: Install Ultralytics
!pip install ultralytics --upgrade
from ultralytics import YOLO
model = YOLO("yolo26m.pt")

model.info(verbose=True)

print(model.model)

import cv2
import matplotlib.pyplot as plt

image = cv2.imread('/content/istockphoto-510399253-612x612.jpg')
result = model(image)

for r in result:
  r.save(filename='result.jpg')

img_result = cv2.imread('result.jpg')
img_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)


from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # downloads automatically on first run

cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()  # draws boxes + labels

    cv2.imshow("YOLO Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # press Q to quit
        break

cap.release()
cv2.destroyAllWindows()

# 
