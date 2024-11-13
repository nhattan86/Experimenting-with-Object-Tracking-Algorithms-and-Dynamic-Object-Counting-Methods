import cv2
from ultralytics import YOLO

# Tải mô hình YOLOv8 đã được huấn luyện
model = YOLO('yolov8n.pt')  # Thay đổi tên mô hình nếu cần

# Khởi tạo video capture
cap = cv2.VideoCapture(0)  # Sử dụng camera mặc định

# Biến để đếm số lượng vật thể
num_objects = 0

# Tọa độ của 2 cạnh hình chữ nhật
line1_x, line1_y = 200, 200  # Cạnh 1 (vào)
line2_x, line2_y = 400, 400  # Cạnh 2 (ra)

# Danh sách để lưu các ID của vật thể
object_ids = []

while True:
    success, frame = cap.read()
    if not success:
        break

    # Dự đoán bằng mô hình YOLO
    results = model(frame)

    # Vẽ các bounding box và đếm vật thể
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()

            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Kiểm tra vật thể đã được đếm chưa
            if int(cls) not in object_ids:
                object_ids.append(int(cls))

                # Kiểm tra vật thể đi qua cạnh 1 và 2
                if x1 < line1_x and x2 > line2_x:
                    num_objects += 1
                elif x1 > line1_x and x2 < line2_x:
                    num_objects -= 1

    # Vẽ 2 cạnh hình chữ nhật
    cv2.line(frame, (line1_x, 0), (line1_x, 480), (0, 255, 0), 2)  # Cạnh 1
    cv2.line(frame, (line2_x, 0), (line2_x, 480), (0, 0, 255), 2)  # Cạnh 2

    # Hiển thị số lượng vật thể đã được đếm
    cv2.putText(frame, f'Num = {num_objects}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị khung hình
    cv2.imshow('YOLOv8 Object Counting', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()