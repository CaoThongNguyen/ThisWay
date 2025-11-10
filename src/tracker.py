import numpy as np
from collections import deque

class SimpleCentroidTracker:
    def __init__(self, max_points=10):
        self.objects = {}
        self.next_object_id = 0
        self.max_points = max_points

    def update(self, current_detections_centroids):
        assigned_ids = {}
        unassigned_detections = []

        for (x, y) in current_detections_centroids:
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

        keys_to_delete = [oid for oid in self.objects if oid not in assigned_ids.values()]
        for key in keys_to_delete:
            del self.objects[key]

        return assigned_ids

    def is_moving(self, object_id, threshold=5):
        if object_id not in self.objects or len(self.objects[object_id]) < self.max_points:
            return True

        history = list(self.objects[object_id])
        first = history[0]
        last = history[-1]
        distance = np.sqrt((last[0] - first[0])**2 + (last[1] - first[1])**2)
        return distance > threshold