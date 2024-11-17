import cv2
import os
import numpy as np
import sys
import importlib.util
from tracker import Tracker
import time
import argparse

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default='0.5')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)

# Set input resolution for webcam
INPUT_WIDTH = 640
INPUT_HEIGHT = 480

# Set display resolution
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Import TensorFlow libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Remove unwanted label if using COCO model
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

# Initialize video stream
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_WIDTH)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_HEIGHT)

# Initialize tracker
tracker = Tracker()

# Define counting lines
cy1 = int(DISPLAY_HEIGHT * 0.2)=168
cy2 = int(DISPLAY_HEIGHT * 0.6)=400
offset = 6

# Initialize counters
counter_up = []
counter_down = []

# Initialize FPS calculation
fps = 0
frame_count = 0
start_time = time.time()

while True:
    ret, frame = video.read()
    if not ret:
        print('Reached the end of the video')
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    detected_objects = []
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            ymin = int(max(1, (boxes[i][0] * DISPLAY_HEIGHT)))
            xmin = int(max(1, (boxes[i][1] * DISPLAY_WIDTH)))
            ymax = int(min(DISPLAY_HEIGHT, (boxes[i][2] * DISPLAY_HEIGHT)))
            xmax = int(min(DISPLAY_WIDTH, (boxes[i][3] * DISPLAY_WIDTH)))
            detected_objects.append([xmin, ymin, xmax, ymax])

    tracked_objects = tracker.update(detected_objects)

    for obj_id, (x1, y1, x2, y2) in tracked_objects.items():
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if cy1 - offset < cy < cy1 + offset:
            if obj_id not in counter_up:
                counter_up.append(obj_id)
        
        if cy2 - offset < cy < cy2 + offset:
            if obj_id not in counter_down:
                counter_down.append(obj_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {obj_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.line(frame, (0, cy1), (DISPLAY_WIDTH, cy1), (0, 0, 255), 2)
    cv2.line(frame, (0, cy2), (DISPLAY_WIDTH, cy2), (255, 0, 0), 2)

    cv2.putText(frame, f'OUT: {len(counter_up)}', (10, 39), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'IN: {len(counter_down)}', (10, 79), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame_count += 1
    if frame_count >= 10:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        frame_count = 0
        start_time = time.time()

    cv2.putText(frame, f'FPS: {fps:.2f}', (DISPLAY_WIDTH - 179, 39), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
