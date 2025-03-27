import os
import cv2
import numpy as np
from ultralytics import YOLO

def normalize_keypoints(bbox, keypoints):
    """Điều chỉnh keypoints theo bbox."""
    xmin, ymin, xmax, ymax = map(int, bbox)
    keypoints_adjusted = keypoints.copy()
    image_size = (xmax - xmin, ymax - ymin)

    for i, (kx, ky) in enumerate(keypoints):
        # Kiểm tra keypoint nằm trong bbox
        if xmin <= kx <= xmax and ymin <= ky <= ymax:
            keypoints_adjusted[i] = [kx - xmin, ky - ymin]
        else:
            keypoints_adjusted[i] = [0, 0]  # Nếu ngoài bbox, gán (0, 0)

    return image_size, keypoints_adjusted

def convert_ratio(new_size, keypoints):
    """Chuyển đổi keypoints thành tỷ lệ (0-1)."""
    w, h = new_size
    keypoints_ratio = []

    for x, y in keypoints:
        # Kiểm tra kích thước hợp lệ và tọa độ dương
        if w > 0 and h > 0 and x >= 0 and y >= 0:
            keypoints_ratio.append([x / w, y / h])
        else:
            keypoints_ratio.append([0, 0])

    return keypoints_ratio


# Load model YOLOv8 Pose
model = YOLO("yolov8m-pose.pt")

# Định nghĩa đường dẫn
input_dir = "F:/PyCharmProjects/DataManipulation/doze_dataset.v7i.yolov8/train/images"
output_csv_path = "dataset/labels.csv"

# Số lượng keypoints (YOLOv8 Pose mặc định là 17 keypoints)
num_keypoints = 17

# Tạo file CSV nếu chưa tồn tại
header = ["class"] + [f"x{i},y{i}" for i in range(num_keypoints)]
if not os.path.exists(output_csv_path):
    with open(output_csv_path, "w") as f:
        f.write(",".join(header) + "\n")

# Xử lý từng ảnh trong thư mục
for file_name in os.listdir(input_dir):
    if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(input_dir, file_name)
    results = model(image_path, conf=0.3, iou=0.5)  # Dự đoán với YOLOv8 Pose

    # Bỏ qua ảnh không phát hiện được người
    if not results[0].keypoints:
        print(f"Không có người trong ảnh {file_name}, bỏ qua.")
        continue

    keypoints = results[0].keypoints.xy.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # Đọc ảnh gốc
    image = cv2.imread(image_path)

    with open(output_csv_path, "a") as f:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])

            # Điều chỉnh keypoints theo bbox
            image_size, person_keypoints = normalize_keypoints(boxes[i], keypoints[i])

            # Chuyển đổi keypoints sang tỷ lệ
            keypoints_ratio = convert_ratio(image_size, person_keypoints)
            print(np.array(keypoints_ratio))
            # Hiển thị ảnh và bounding box
            temp_image = image.copy()
            cv2.rectangle(temp_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.imshow("Object", temp_image)
            key = cv2.waitKey(0)  # Nhấn phím nhập class
            cv2.destroyAllWindows()

            if key == 27:  # ESC để bỏ qua đối tượng
                continue

            # Xác định class: Giới hạn phím chữ và số (48-57: 0-9, 97-122: a-z)
            if (48 <= key <= 57) or (97 <= key <= 122):
                class_id = chr(key)
            else:
                print("Phím không hợp lệ, bỏ qua đối tượng.")
                continue

            # Chuẩn bị dữ liệu cho CSV
            row = [class_id]
            for j in range(num_keypoints):
                if j < len(keypoints_ratio):
                    x_norm, y_norm = keypoints_ratio[j]
                    row.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])
                else:
                    row.extend(["0", "0"])

            # Ghi vào CSV
            f.write(",".join(row) + "\n")

    print(f"Đã xử lý {file_name}.")

print(f"Lưu annotations vào {output_csv_path}")
print("Hoàn thành!")
