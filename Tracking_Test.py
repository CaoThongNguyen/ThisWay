import cv2
import torch
import numpy as np
from torchvision.ops import box_iou
from collections import deque


class SimpleCentroidTracker:
    def __init__(self, max_points=10):
        # Dictionary: { object_id: deque([centroid_1, centroid_2, ...], maxlen=max_points) }
        self.objects = {}
        self.next_object_id = 0
        self.max_points = max_points

    def update(self, current_detections_centroids):

        assigned_ids = {}
        unassigned_detections = []

        for (x, y) in current_detections_centroids:
            is_assigned = False
            best_match_id = -1
            min_distance = float('inf')

            for object_id, centroids_history in self.objects.items():
                last_centroid = centroids_history[-1]
                distance = np.sqrt((x - last_centroid[0])**2 + (y - last_centroid[1])**2)

                if distance < 50 and distance < min_distance:
                    min_distance = distance
                    best_match_id = object_id

            if best_match_id != -1:
                self.objects[best_match_id].append((x, y))
                assigned_ids[(x, y)] = best_match_id
            else:
                unassigned_detections.append((x, y))

        for (x, y) in unassigned_detections:
            self.objects[self.next_object_id] = deque([(x, y)], maxlen=self.max_points)
            assigned_ids[(x, y)] = self.next_object_id
            self.next_object_id += 1

        keys_to_delete = []
        for object_id in self.objects.keys():
            if object_id not in assigned_ids.values():
                keys_to_delete.append(object_id)
        for key in keys_to_delete:
            del self.objects[key]

        return assigned_ids

    def is_moving(self, object_id, threshold=5):
        """Kiểm tra xem một đối tượng có di chuyển đáng kể hay không."""
        if object_id not in self.objects or len(self.objects[object_id]) < self.max_points:
            return True # Coi là di chuyển nếu lịch sử chưa đủ dài

        history = list(self.objects[object_id])
        first_centroid = history[0]
        last_centroid = history[-1]

        distance = np.sqrt((last_centroid[0] - first_centroid[0])**2 + (last_centroid[1] - first_centroid[1])**2)

        return distance > threshold


priority_weights = {
    0: 0,  # person
    1: 1,  # bicycle
    2: 3,  # car
    3: 2,  # motorcycle
    5: 5,  # bus
    7: 4   # truck
}

TARGET_CLASSES = list(priority_weights.keys())

print("Đang tải mô hình YOLOv5m...")
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    model.conf = 0.45
    model.classes = TARGET_CLASSES
    print("Tải mô hình hoàn tất.")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    exit()

video_path = "C:/Users/LENOVO/Downloads/videoplayback.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không mở được video. Vui lòng kiểm tra lại đường dẫn.")
    exit()

print("Bắt đầu xử lý video. Nhấn 'q' để thoát.")

tracker = SimpleCentroidTracker(max_points=10) # Theo dõi 10 khung hình (~ 0.3-0.5 giây)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, size=360)

    # Khởi tạo biến đếm và điểm ưu tiên cho frame này
    total_priority_score = 0
    vehicle_counts = {}

    detections = results.xyxy[0]
    detections_np = detections.cpu().numpy()

    final_detections = []
    ignored_indices = set()


    current_centroids = []
    # Lưu trữ thông tin phát hiện theo thứ tự Centroid để Tracker có thể gán ID
    detection_map = {}

    for i, det in enumerate(detections_np):
        x1, y1, x2, y2 = map(int, det[:4])
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2
        centroid = (centroid_x, centroid_y)

        current_centroids.append(centroid)
        detection_map[centroid] = det # Lưu lại toàn bộ detection

    assigned_ids = tracker.update(current_centroids)

    moving_detections_np = []

    # Lọc các đối tượng đang đứng yên
    for centroid, obj_id in assigned_ids.items():
        if tracker.is_moving(obj_id):
            moving_detections_np.append(detection_map[centroid])

    # Nếu không có đối tượng di chuyển nào được phát hiện, chuyển sang detections thô (để không bị lỗi)
    if len(moving_detections_np) == 0:
        moving_detections_np = detections_np
    else:
        # *** SỬA LỖI TẠI ĐÂY ***
        # Nếu CÓ đối tượng di chuyển, nó đang là list, cần chuyển sang numpy array
        moving_detections_np = np.array(moving_detections_np)


    # LIÊN KẾT NGƯỜI VÀ PHƯƠNG TIỆN NHỎ (TRÊN CÁC ĐỐI TƯỢNG ĐANG DI CHUYỂN)

    # Lấy các detections của Người (Class 0)
    person_indices = np.where(moving_detections_np[:, 5] == 0)[0]
    person_boxes = moving_detections_np[person_indices]

    # Lấy các detections của Xe máy (Class 3) và Xe đạp (Class 1)
    small_vehicle_indices = np.where((moving_detections_np[:, 5] == 3) | (moving_detections_np[:, 5] == 1))[0]
    small_vehicle_boxes = moving_detections_np[small_vehicle_indices]

    if len(person_boxes) > 0 and len(small_vehicle_boxes) > 0:
        # Chuyển đổi sang Tensor để tính IOU
        person_tensor = torch.from_numpy(person_boxes[:, :4])
        vehicle_tensor = torch.from_numpy(small_vehicle_boxes[:, :4])
        iou_matrix = box_iou(person_tensor, vehicle_tensor)

        for i in range(iou_matrix.shape[0]):
            for j in range(iou_matrix.shape[1]):
                if iou_matrix[i, j] > 0.3:
                    final_detections.append(small_vehicle_boxes[j])

                    # Cần ánh xạ lại index của detection đã dùng để bỏ qua
                    person_det = person_boxes[i]
                    vehicle_det = small_vehicle_boxes[j]

                    # Tìm index tương ứng trong Moving Detections để bỏ qua
                    p_idx = np.where(np.all(moving_detections_np == person_det, axis=1))[0][0]
                    v_idx = np.where(np.all(moving_detections_np == vehicle_det, axis=1))[0][0]

                    ignored_indices.add(p_idx)
                    ignored_indices.add(v_idx)
                    break

                    # --- BƯỚC 2: THÊM CÁC ĐỐI TƯỢNG CÒN LẠI VÀO DANH SÁCH CUỐI CÙNG ---

    for i, detection in enumerate(moving_detections_np):
        if i in ignored_indices:
            continue
        final_detections.append(detection)

    # ----------------------------------------------------
    # BƯỚC C: TÍNH TOÁN VÀ VẼ KẾT QUẢ CUỐI CÙNG
    # ----------------------------------------------------

    for detection in final_detections:
        x1, y1, x2, y2 = map(int, detection[:4])
        conf = detection[4]
        cls_id = int(detection[5])

        label = model.names.get(cls_id, 'Unknown')
        weight = priority_weights.get(cls_id, 0)

        if cls_id != 0:
            total_priority_score += weight

        vehicle_counts[label] = vehicle_counts.get(label, 0) + 1

        # --- Vẽ Bounding Boxes và Text ---
        color = (0, 255, 0)
        if cls_id in [5]: color = (255, 0, 0)
        elif cls_id in [7]: color = (0, 0, 255)
        elif cls_id in [2]: color = (0, 165, 255)

        # Vẽ ID nếu cần thiết để kiểm tra tracker
        # cv2.putText(frame, f'ID: {obj_id}', (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} ({weight}) {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- 5. HIỂN THỊ THÔNG SỐ TỔNG HỢP ---

    display_score = f"TOTAL PRIORITY SCORE (MOVING): {total_priority_score}"
    cv2.putText(frame, display_score, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    y_offset = 70
    for label, count in vehicle_counts.items():
        weight = priority_weights.get(next((k for k, v in model.names.items() if v == label), 0), 0)
        count_text = f"| {label}s (Moving): {count} (Weight: {weight})"
        cv2.putText(frame, count_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        y_offset += 30

    cv2.imshow("Smart Traffic System Detection (Moving Filtered)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Kết thúc xử lý video và đóng cửa sổ.")