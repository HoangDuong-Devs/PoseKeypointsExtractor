import cv2
import os
import torch
from ultralytics import YOLO
import numpy as np
# Load model YOLOv8 Pose
model = YOLO("yolov8m-pose.pt")

# Đọc ảnh và chạy mô hình
image_path = "image.png"
results = model(image_path, conf=0.4)

# Duyệt qua từng đối tượng được phát hiện
for result in results:
    keypoints = result.keypoints.xyn  # Lấy (x, y)
    
    print(np.array(keypoints))
    
    confidence = result.keypoints.conf  # Lấy độ tin cậy (confidence)
    # Chuyển sang NumPy để dễ xử lý
    keypoints_np = keypoints.cpu().numpy() if isinstance(keypoints, torch.Tensor) else keypoints
    confidence_np = confidence.cpu().numpy() if isinstance(confidence, torch.Tensor) else confidence


# Vẽ keypoints lên ảnh
image_with_keypoints = results[0].plot()
image_with_keypoints = cv2.resize(image_with_keypoints, (640, 640))

cv2.imshow("Pose Estimation", image_with_keypoints)
cv2.waitKey(0)  # Đợi nhấn phím bất kỳ để đóng cửa sổ
cv2.destroyAllWindows()
