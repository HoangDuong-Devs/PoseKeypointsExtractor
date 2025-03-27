import os
import cv2
import numpy as np
from ultralytics import YOLO

def normalize_keypoints(bbox, keypoints):
    """Điều chỉnh keypoints theo bbox."""
    xmin, ymin, xmax, ymax = map(int, bbox)
    keypoints_adjusted = keypoints.copy()
    print("Keypoints: ", keypoints)
    # Kiểm tra và chuẩn hóa tọa độ keypoints
    for i, (kx, ky) in enumerate(keypoints):
        if xmin <= kx <= xmax and ymin <= ky <= ymax:
            keypoints_adjusted[i] = [kx - xmin, ky - ymin]
        else:
            # Nếu keypoint nằm ngoài bbox, đặt về (0, 0)
            keypoints_adjusted[i] = [0, 0]

    image_size = (xmax - xmin, ymax - ymin)
    return image_size, keypoints_adjusted

def convert_ratio(new_size, keypoints):
    """Chuyển đổi keypoints thành tỷ lệ."""
    w, h = new_size
    keypoints_ratio = keypoints.copy()

    for i, (x, y) in enumerate(keypoints):
        # Đảm bảo tọa độ không âm và kích thước hợp lệ
        if x >= 0 and y >= 0 and w > 0 and h > 0:
            keypoints_ratio[i] = [x / w, y / h]
        else:
            keypoints_ratio[i] = [0, 0]

    return keypoints_ratio


# Load model YOLOv8 Pose
model = YOLO("yolov8m-pose.pt")

# Định nghĩa thư mục đầu vào và đường dẫn file CSV đầu ra
input_dir = "F:/PyCharmProjects/DataManipulation/doze_dataset.v7i.yolov8/train/images"

output_csv_path = "dataset/labels.csv"

# Số lượng keypoints (YOLOv8 Pose mặc định là 17 keypoints)
num_keypoints = 17

# Xác định header file CSV
header = ["class"] + [f"x{i},y{i}" for i in range(num_keypoints)]

# Kiểm tra nếu file chưa tồn tại thì tạo mới với header
if not os.path.exists(output_csv_path):
    with open(output_csv_path, "w") as f:
        f.write(",".join(header) + "\n")

# Bắt đầu xử lý ảnh
for file_name in os.listdir(input_dir):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_dir, file_name)
        results = model(image_path, conf=0.3, iou=0.5)  # Chạy mô hình

        if results[0].keypoints is None:
            print(f"Không có người trong ảnh {file_name}, bỏ qua.")
            continue

        # Trích xuất keypoints và bounding boxes
        keypoints = results[0].keypoints.xy.cpu().numpy()  # Lấy keypoints (x, y)

        boxes = results[0].boxes.xyxy.cpu().numpy()  # Lấy tọa độ bounding box
        # Đọc ảnh để lấy kích thước
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        with open(output_csv_path, "a") as f:  # Mở file CSV ở chế độ append
            for i in range(len(boxes)):  # Duyệt từng người phát hiện
                temp_image = image.copy()
                x1, y1, x2, y2 = map(int, boxes[i])  # Lấy tọa độ bbox
                image_size, person_keypoints = normalize_keypoints(boxes[i], keypoints[i])
                # Chuyển đổi keypoints sang tỷ lệ sau khi xoay ảnh
                keypoints_ratio = convert_ratio(image_size, person_keypoints)
                print(keypoints_ratio)

                # Vẽ bounding box
                cv2.rectangle(temp_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # Hiển thị từng đối tượng để nhập class
                cv2.imshow("Object", temp_image)
                key = cv2.waitKey(0)  # Chờ nhập phím
                cv2.destroyAllWindows()

                if key == 27:  # Nhấn ESC để bỏ qua đối tượng này
                    continue
                class_id = chr(key)  # Chuyển phím nhập thành class

                # Chuẩn bị dữ liệu cho CSV
                row = [class_id]
                for j in range(num_keypoints):
                    if j < len(keypoints):
                        x_norm, y_norm = keypoints_ratio[j]
                        row.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])
                    else:
                        row.extend(["0", "0"])

                # Ghi trực tiếp vào CSV
                f.write(",".join(row) + "\n")

        print(f"Đã xử lý {file_name}.")

print(f"Lưu annotations vào {output_csv_path}")
print("Hoàn thành!")
