import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')  # Load the pre-trained YOLOv8 model

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
# - max_age: maximum number of frames a track will be kept alive without being confirmed
# - n_init: minimum number of consecutive frames required to be tracked before confirmed
# - nms_max_overlap: maximum allowed overlap between detections for non-max suppression

# Open video (replace '' with your video path)
video_path = ''
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video")

# Set up video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_filename = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_filename, fourcc, 20, (frame_width, frame_height))

frame_count = 0

# Define two vertical lines for counting
DISPLAY_WIDTH = frame_width
DISPLAY_HEIGHT = frame_height

line1_x = int(DISPLAY_WIDTH * 0.4)  # Line 1 (example: left)
line2_x = int(DISPLAY_WIDTH * 0.5)  # Line 2 (example: right)

# Initialize counters
counter_in = 0
counter_out = 0

# Create dictionary to store object information
object_info = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model to detect objects
    results = model(frame)

    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Only process detections with class 0 (person) and confidence > 0.5
                if cls_id == 0 and conf > 0.5:
                    # DeepSORT uses bbox format: [x_min, y_min, width, height]
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    detections.append((bbox, conf, 'person', None))

    # Update tracker with detections
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Calculate center of bounding box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Store or update object information
        if track_id not in object_info:
            object_info[track_id] = {'positions': [(cx, cy)], 'lines_crossed': [], 'counted_in': False, 'counted_out': False}
        else:
            object_info[track_id]['positions'].append((cx, cy))
            if len(object_info[track_id]['positions']) > 2:
                object_info[track_id]['positions'].pop(0)

            prev_cx, prev_cy = object_info[track_id]['positions'][0]

            # Check if object crossed any vertical line
            if (prev_cx <= line1_x and cx > line1_x) or (prev_cx >= line1_x and cx < line1_x):
                if 'line1' not in object_info