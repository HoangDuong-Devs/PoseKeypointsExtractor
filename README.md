# process_keypoints_PE_csv.py

## Giới thiệu

Script `process_keypoints_PE_csv.py` được thiết kế để:
- Sử dụng mô hình YOLOv8 Pose để phát hiện keypoints từ hình ảnh.
- Chuẩn hóa keypoints theo bounding box của từng đối tượng.
- Chuyển đổi keypoints sang tỷ lệ tương đối.
- Xuất dữ liệu keypoints ra file CSV, kèm theo nhãn (class) của từng đối tượng.

## Yêu cầu hệ thống

- Python 3.12
- Ubuntu 24.04 hoặc Windows

## Cài đặt

1. Tạo môi trường ảo (tuỳ chọn):

```bash
python -m venv venv
source venv/bin/activate  # Linux
venv\Scripts\activate  # Windows
```

2. Cài đặt các thư viện cần thiết:

```bash
pip install ultralytics opencv-python numpy
```

## Hướng dẫn sử dụng

### 1. Cấu trúc thư mục

```
.
├── process_keypoints_PE_csv.py
├── images/
│    ├──  image1.jpg
│    ├──  image2.jpg
│    └──  ...
└── dataset/
      └── labels.csv
```

### 2. Chạy script

```bash
python process_keypoints_PE_csv.py
```

### 3. Quy trình hoạt động

1. **Đọc ảnh đầu vào:** Tất cả các ảnh trong thư mục `images`.
2. **Dự đoán keypoints:** Sử dụng mô hình YOLOv8 Pose (`yolov8m-pose.pt`).
3. **Chuẩn hóa keypoints:** Điều chỉnh keypoints theo bounding box của đối tượng.
4. **Gán nhãn (class) thủ công:** Hiển thị từng đối tượng, nhấn phím tương ứng để gán nhãn.
   - Nhấn `ESC` để bỏ qua đối tượng.
   - Nhấn bất kỳ phím nào khác để gán nhãn tương ứng.
5. **Ghi dữ liệu ra CSV:** Dữ liệu được lưu vào `dataset/labels.csv`.

### 4. Định dạng file CSV

File CSV đầu ra có định dạng như sau:

```
class,x0,y0,x1,y1,...,x16,y16
A,0.125,0.250,0.375,...,0.875
B,0.150,0.300,0.400,...,0.900
```

- **class:** Nhãn của đối tượng do người dùng nhập.
- **x, y:** Tọa độ keypoints theo tỷ lệ (giá trị từ 0 đến 1).

## Tuỳ chỉnh

1. **Thay đổi mô hình YOLOv8 Pose:**
   - Thay đổi dòng này để sử dụng model khác:

```python
model = YOLO("yolov8m-pose.pt")
```

2. **Chỉnh ngưỡng phát hiện:**

```python
results = model(image_path, conf=0.3, iou=0.5)
```

- `conf`: Ngưỡng tin cậy (mặc định: 0.3).
- `iou`: Ngưỡng IoU (mặc định: 0.5).

## Lưu ý

- Nhấn `ESC` để bỏ qua đối tượng không cần lưu.
- Đảm bảo file `yolov8m-pose.pt` được tải xuống và đặt đúng vị trí.

## Tài liệu tham khảo

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)

