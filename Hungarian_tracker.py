import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment  # For Hungarian Algorithm

class Tracker:
    def __init__(self, max_distance=50, max_disappeared=10):
        self.tracked_objects = {}  # Stores objects with IDs
        self.disappeared = {}  # Tracks how long objects have been missing
        self.max_distance = max_distance  # Maximum distance for matching
        self.max_disappeared = max_disappeared  # Max frames before an object is removed
        self.next_object_id = 1  # Counter for unique IDs

    def update(self, new_rectangles):
        if len(self.tracked_objects) == 0:
            # Initialize with new rectangles
            for rect in new_rectangles:
                self.tracked_objects[self.next_object_id] = rect
                self.disappeared[self.next_object_id] = 0
                self.next_object_id += 1
            return self.tracked_objects

        # Match existing objects to new detections using Hungarian Algorithm
        object_ids = list(self.tracked_objects.keys())
        object_rects = list(self.tracked_objects.values())
        
        # Compute cost matrix
        cost_matrix = np.zeros((len(object_rects), len(new_rectangles)))
        for i, obj_rect in enumerate(object_rects):
            for j, new_rect in enumerate(new_rectangles):
                obj_center = (
                    (obj_rect[0] + obj_rect[2]) / 2,
                    (obj_rect[1] + obj_rect[3]) / 2,
                )
                new_center = (
                    (new_rect[0] + new_rect[2]) / 2,
                    (new_rect[1] + new_rect[3]) / 2,
                )
                cost_matrix[i, j] = np.linalg.norm(np.array(obj_center) - np.array(new_center))

        # Apply Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Track matched objects
        matched_ids = set()
        for row, col in zip(row_ind, col_ind):
            if cost_matrix[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.tracked_objects[object_id] = new_rectangles[col]
            self.disappeared[object_id] = 0
            matched_ids.add(object_id)

        # Handle unmatched objects
        unmatched_object_ids = set(object_ids) - matched_ids
        for object_id in unmatched_object_ids:
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                del self.tracked_objects[object_id]
                del self.disappeared[object_id]

        # Handle new detections
        unmatched_detections = set(range(len(new_rectangles))) - set(col_ind)
        for index in unmatched_detections:
            self.tracked_objects[self.next_object_id] = new_rectangles[index]
            self.disappeared[self.next_object_id] = 0
            self.next_object_id += 1

        return self.tracked_objects

# Example usage
if __name__ == "__main__":
    tracker = Tracker(max_distance=100)

    # Simulate a sequence of detections
    detections_frame_1 = [(10, 10, 50, 50), (200, 200, 250, 250)]
    detections_frame_2 = [(15, 15, 55, 55), (205, 205, 255, 255)]
    detections_frame_3 = [(20, 20, 60, 60)]

    print("Frame 1:", tracker.update(detections_frame_1))
    print("Frame 2:", tracker.update(detections_frame_2))
    print("Frame 3:", tracker.update(detections_frame_3))
