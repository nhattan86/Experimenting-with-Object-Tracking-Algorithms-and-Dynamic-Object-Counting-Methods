import argparse
import cv2
import os
import numpy as np
import sys
import importlib.util
from tracker import Tracker
import time

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.5)
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

# Set input resolution for webcam
INPUT_WIDTH = 860
INPUT_HEIGHT = 790

# Set display resolution
DISPLAY_WIDTH = 860
DISPLAY_HEIGHT = 790

# Import TensorFlow libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize video stream
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_WIDTH)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_HEIGHT)

# Initialize tracker
tracker = Tracker()

# # Define counting lines (adjust these values as needed)
# cy1 = int(DISPLAY_HEIGHT * 0.2)  # Upper line at % of the frame height
# cy2 = int(DISPLAY_HEIGHT * 0.6)  # Lower line at % of the frame height
# offset = 6

# Define virtual rectangle
rect_width = int(DISPLAY_WIDTH * 0.8)
rect_height = int(DISPLAY_HEIGHT * 0.4)
rect_x = (DISPLAY_WIDTH - rect_width) // 2
rect_y = (DISPLAY_HEIGHT - rect_height) // 2
a = rect_y
b = rect_x
c = rect_x + rect_width
d = rect_y + rect_height

# Initialize counters
counter_up = []
counter_down = []
total = 0

# Initialize FPS calculation
fps = 0
frame_count = 0
start_time = time.time()

# Dictionary to store object directions
object_directions = {}

while True:
    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
        print('Reached the end of the video')
        break
    
    # Resize frame to display resolution
    #frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    detected_objects = []
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            ymin = int(max(1, (boxes[i][0] * DISPLAY_HEIGHT)))
            xmin = int(max(1, (boxes[i][1] * DISPLAY_WIDTH)))
            ymax = int(min(DISPLAY_HEIGHT, (boxes[i][2] * DISPLAY_HEIGHT)))
            xmax = int(min(DISPLAY_WIDTH, (boxes[i][3] * DISPLAY_WIDTH)))
            detected_objects.append([xmin, ymin, xmax, ymax])

    # Update tracker
    tracked_objects = tracker.update(detected_objects)

    # Loop through tracked objects for counting
    for obj_id, (x1, y1, x2, y2) in tracked_objects.items():
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw yellow dot at the center of the object
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

        # Update object direction
        if obj_id in object_directions:
            prev_y = object_directions[obj_id]
            if cy < prev_y:
                direction = 'up'
            elif cy > prev_y:
                direction = 'down'
            else:
                direction = 'stationary'
            
            # Update TOTAL
            if direction == 'up' and prev_y > d and cy < a:
                total += 1
            elif direction == 'down' and prev_y < a and cy > d:
                total -= 1
        else:
            direction = 'new'
        
        object_directions[obj_id] = cy

        # Update IN and OUT counters
        if a < cy < d:
            if obj_id not in counter_up and obj_id not in counter_down:
                if cx < b:
                    counter_up.append(obj_id)
                elif cx > c:
                    counter_down.append(obj_id)

        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {obj_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw virtual rectangle
    cv2.rectangle(frame, (b, a), (c, d), (255, 0, 0), 2)

    # Display counts
    cv2.putText(frame, f'OUT: {len(counter_up)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'IN: {len(counter_down)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'TOTAL: {total}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Calculate and display FPS
    frame_count += 1
    if frame_count >= 10:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        frame_count = 0
        start_time = time.time()

    cv2.putText(frame, f'FPS: {fps:.2f}', (DISPLAY_WIDTH - 239, 39), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
