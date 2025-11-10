import cv2
import numpy as np
from .tracker import SimpleCentroidTracker
from .utils import (
    load_model, filter_moving_objects, link_person_vehicle,
    draw_results, PRIORITY_WEIGHTS
)


class TrafficDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.tracker = SimpleCentroidTracker(max_points=10)

    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, size=360)
        detections_np = results.xyxy[0].cpu().numpy()

        if len(detections_np) == 0:
            return frame, 0, {}

        moving_detections = filter_moving_objects(detections_np, self.tracker)
        final_detections = link_person_vehicle(moving_detections)

        total_score = 0
        vehicle_counts = {}
        for det in final_detections:
            cls_id = int(det[5])
            if cls_id != 0:
                weight = PRIORITY_WEIGHTS.get(cls_id, 0)
                total_score += weight
                label = self.model.names[cls_id]
                vehicle_counts[label] = vehicle_counts.get(label, 0) + 1

        frame = draw_results(frame, final_detections, self.model, total_score, vehicle_counts)

        return frame, total_score, vehicle_counts