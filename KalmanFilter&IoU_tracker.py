import numpy as np
import cv2
from filterpy.kalman import KalmanFilter

def iou(boxA, boxB):
    """Calculate the Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class KalmanTracker:
    """Tracker with Kalman Filter for prediction and IoU for matching."""

    def __init__(self, max_lost=10, iou_threshold=0.3):
        self.trackers = {}  # {id: {'kf': KalmanFilter, 'bbox': [x1, y1, x2, y2], 'lost': int}}
        self.next_id = 1  # ID counter for new objects
        self.max_lost = max_lost  # Maximum frames an object can be lost
        self.iou_threshold = iou_threshold  # Threshold for IoU matching

    def _create_kalman_filter(self, bbox):
        """Initialize a Kalman Filter for tracking."""
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])
        kf.R[2:, 2:] *= 10.  # Measurement noise
        kf.P[4:, 4:] *= 1000.  # State uncertainty
        kf.P *= 10.  # Uncertainty
        kf.Q[-1, -1] *= 0.01  # Process noise
        kf.Q[4:, 4:] *= 0.01
        kf.x[:4] = np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]]])
        return kf

    def _predict(self):
        """Predict the next position of all tracked objects."""
        for obj_id in list(self.trackers.keys()):
            tracker = self.trackers[obj_id]
            tracker['kf'].predict()
            state = tracker['kf'].x.flatten()
            tracker['bbox'] = [state[0], state[1], state[2], state[3]]

    def update(self, detections):
        """Update the tracker with new detections."""
        self._predict()

        # Matching current detections with tracked objects using IoU
        matches = []
        unmatched_detections = set(range(len(detections)))
        unmatched_trackers = set(self.trackers.keys())

        if detections:
            iou_matrix = np.zeros((len(self.trackers), len(detections)), dtype=np.float32)

            for i, (obj_id, tracker) in enumerate(self.trackers.items()):
                for j, detection in enumerate(detections):
                    iou_matrix[i, j] = iou(tracker['bbox'], detection)

            iou_indices = np.argwhere(iou_matrix > self.iou_threshold)
            for i, j in iou_indices:
                obj_id = list(self.trackers.keys())[i]
                if i in unmatched_trackers and j in unmatched_detections:
                    matches.append((obj_id, j))
     
