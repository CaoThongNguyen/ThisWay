import cv2
import numpy as np
from torchvision.ops import box_iou
import torch

PRIORITY_WEIGHTS = {
    0: 0,  # person
    1: 1,  # bicycle
    2: 3,  # car
    3: 2,  # motorcycle
    5: 5,  # bus
    7: 4  # truck
}

TARGET_CLASSES = list(PRIORITY_WEIGHTS.keys())


def load_model(model_path, conf=0.45):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, source='local')
    model.conf = conf
    model.classes = TARGET_CLASSES
    return model


def filter_moving_objects(detections_np, tracker):
    current_centroids = []
    detection_map = {}

    for i, det in enumerate(detections_np):
        x1, y1, x2, y2 = map(int, det[:4])
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        current_centroids.append(centroid)
        detection_map[centroid] = det

    assigned_ids = tracker.update(current_centroids)
    moving_detections = []

    for centroid, obj_id in assigned_ids.items():
        if tracker.is_moving(obj_id):
            moving_detections.append(detection_map[centroid])

    return np.array(moving_detections) if moving_detections else detections_np


def link_person_vehicle(detections_np):
    final_detections = []
    ignored_indices = set()

    person_indices = np.where(detections_np[:, 5] == 0)[0]
    person_boxes = detections_np[person_indices]

    small_vehicle_indices = np.where((detections_np[:, 5] == 3) | (detections_np[:, 5] == 1))[0]
    small_vehicle_boxes = detections_np[small_vehicle_indices]

    if len(person_boxes) > 0 and len(small_vehicle_boxes) > 0:
        person_tensor = torch.from_numpy(person_boxes[:, :4])
        vehicle_tensor = torch.from_numpy(small_vehicle_boxes[:, :4])
        iou_matrix = box_iou(person_tensor, vehicle_tensor)

        for i in range(iou_matrix.shape[0]):
            for j in range(iou_matrix.shape[1]):
                if iou_matrix[i, j] > 0.3:
                    final_detections.append(small_vehicle_boxes[j])

                    p_idx = np.where(np.all(detections_np == person_boxes[i], axis=1))[0][0]
                    v_idx = np.where(np.all(detections_np == small_vehicle_boxes[j], axis=1))[0][0]
                    ignored_indices.add(p_idx)
                    ignored_indices.add(v_idx)
                    break

    for i, detection in enumerate(detections_np):
        if i not in ignored_indices:
            final_detections.append(detection)

    return np.array(final_detections)


def draw_results(frame, final_detections, model, total_score, vehicle_counts):
    colors = {
        5: (255, 0, 0),  # bus
        7: (0, 0, 255),  # truck
        2: (0, 165, 255),  # car
    }

    for detection in final_detections:
        x1, y1, x2, y2 = map(int, detection[:4])
        conf, cls_id = detection[4], int(detection[5])

        label = model.names[cls_id]
        weight = PRIORITY_WEIGHTS.get(cls_id, 0)
        color = colors.get(cls_id, (0, 255, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} ({weight}) {conf:.2f}',
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"TOTAL PRIORITY SCORE: {total_score}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    y_offset = 70
    for label, count in vehicle_counts.items():
        weight = PRIORITY_WEIGHTS.get(next((k for k, v in model.names.items() if v == label), 0), 0)
        text = f"| {label}s: {count} (Weight: {weight})"
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        y_offset += 30

    return frame