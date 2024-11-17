import argparse
import cv2
import os
import numpy as np
import sys
import importlib.util
from tracker import Tracker
import time
from collections import deque
import datetime

class ObjectDetector:
    def __init__(self, args):
        # Configuration
        self.MODEL_NAME = args.modeldir
        self.GRAPH_NAME = args.graph
        self.LABELMAP_NAME = args.labels
        self.min_conf_threshold = float(args.threshold)
        self.use_TPU = args.edgetpu
        
        # Display settings
        self.INPUT_WIDTH = 1280  # Increased resolution
        self.INPUT_HEIGHT = 720  # 720p format
        self.DISPLAY_WIDTH = 1280
        self.DISPLAY_HEIGHT = 720
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)  # Store FPS history for smoothing
        self.frame_count = 0
        self.start_time = time.time()
        
        # Tracking state
        self.counter_up = []
        self.counter_down = []
        self.total = 0
        self.object_directions = {}
        self.object_trails = {}  # Store motion trails
        
        self._initialize_detector()
        self._initialize_video()
        self._setup_visualization()
        
    def _initialize_detector(self):
        # Initialize TensorFlow Lite interpreter
        pkg = importlib.util.find_spec('tflite_runtime')
        if pkg:
            from tflite_runtime.interpreter import Interpreter
            if self.use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if self.use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # Load model and labels
        CWD_PATH = os.getcwd()
        PATH_TO_CKPT = os.path.join(CWD_PATH, self.MODEL_NAME, self.GRAPH_NAME)
        PATH_TO_LABELS = os.path.join(CWD_PATH, self.MODEL_NAME, self.LABELMAP_NAME)

        with open(PATH_TO_LABELS, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        if self.labels[0] == '???':
            del(self.labels[0])

        if self.use_TPU:
            self.interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                        experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        else:
            self.interpreter = Interpreter(model_path=PATH_TO_CKPT)

        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.floating_model = (self.input_details[0]['dtype'] == np.float32)
        self.input_mean = 127.5
        self.input_std = 127.5

    def _initialize_video(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, self.INPUT_WIDTH)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, self.INPUT_HEIGHT)
        self.video.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
        self.video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG codec
        self.tracker = Tracker()

    def _setup_visualization(self):
        # Define monitoring area (rectangle)
        self.rect_width = int(self.DISPLAY_WIDTH * 0.7)  # Slightly smaller
        self.rect_height = int(self.DISPLAY_HEIGHT * 0.5)
        self.rect_x = (self.DISPLAY_WIDTH - self.rect_width) // 2
        self.rect_y = (self.DISPLAY_HEIGHT - self.rect_height) // 2
        
        # Rectangle boundaries
        self.a = self.rect_y
        self.b = self.rect_x
        self.c = self.rect_x + self.rect_width
        self.d = self.rect_y + self.rect_height
        
        # Colors
        self.COLORS = {
            'rectangle': (255, 165, 0),  # Orange
            'text': (255, 255, 255),  # White
            'trail': (0, 255, 255),  # Yellow
            'bbox': (50, 205, 50),   # Lime Green
            'counter_bg': (0, 0, 0)   # Black
        }

    def _process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        return (
            self.interpreter.get_tensor(self.output_details[0]['index'])[0],
            self.interpreter.get_tensor(self.output_details[1]['index'])[0],
            self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        )

    def _draw_interface(self, frame):
        # Add blur effect to background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT), 
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

        # Draw monitoring area with gradient
        cv2.rectangle(frame, (self.b, self.a), (self.c, self.d), 
                     self.COLORS['rectangle'], 2)
        
        # Draw counter background
        counter_height = 140
        cv2.rectangle(frame, (0, 0), (250, counter_height), 
                     self.COLORS['counter_bg'], -1)
        
        # Add counters with enhanced styling
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'OUT: {len(self.counter_up)}', (20, 40), font, 
                   1, self.COLORS['text'], 2)
        cv2.putText(frame, f'IN: {len(self.counter_down)}', (20, 80), font,
                   1, self.COLORS['text'], 2)
        cv2.putText(frame, f'TOTAL: {self.total}', (20, 120), font,
                   1, self.COLORS['text'], 2)

        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (self.DISPLAY_WIDTH - 200, self.DISPLAY_HEIGHT - 20),
                   font, 0.5, self.COLORS['text'], 1)

        # Calculate and display smooth FPS
        self.frame_count += 1
        if self.frame_count >= 30:
            end_time = time.time()
            fps = self.frame_count / (end_time - self.start_time)
            self.fps_history.append(fps)
            self.frame_count = 0
            self.start_time = time.time()

        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        cv2.putText(frame, f'FPS: {avg_fps:.1f}', (self.DISPLAY_WIDTH - 150, 30),
                   font, 1, self.COLORS['text'], 2)

        return frame

    def _update_tracking(self, detected_objects, frame):
        tracked_objects = self.tracker.update(detected_objects)
        
        for obj_id, (x1, y1, x2, y2) in tracked_objects.items():
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Update motion trails
            if obj_id not in self.object_trails:
                self.object_trails[obj_id] = deque(maxlen=20)
            self.object_trails[obj_id].append((cx, cy))
            
            # Draw motion trails
            points = list(self.object_trails[obj_id])
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], self.COLORS['trail'], 2)
            
            # Update direction and counting
            if obj_id in self.object_directions:
                prev_y = self.object_directions[obj_id]
                if cy < prev_y and prev_y > self.d and cy < self.a:
                    self.total += 1
                elif cy > prev_y and prev_y < self.a and cy > self.d:
                    self.total -= 1
            
            self.object_directions[obj_id] = cy
            
            # Update counters
            if self.a < cy < self.d:
                if obj_id not in self.counter_up and obj_id not in self.counter_down:
                    if cx < self.b:
                        self.counter_up.append(obj_id)
                    elif cx > self.c:
                        self.counter_down.append(obj_id)
            
            # Draw bounding box and ID with enhanced styling
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLORS['bbox'], 2)
            label = f'ID: {obj_id}'
            
            # Add label background
            (label_width, label_height), _ = cv2.getFont(cv2.FONT_HERSHEY_SIMPLEX).getTextSize(
                label, 1, 2)
            cv2.rectangle(frame, (x1, y1-30), (x1 + label_width, y1),
                         self.COLORS['bbox'], -1)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, self.COLORS['text'], 2)

    def run(self):
        try:
            while True:
                ret, frame = self.video.read()
                if not ret:
                    print('Failed to grab frame')
                    break

                # Process frame
                boxes, classes, scores = self._process_frame(frame)
                
                # Detect objects
                detected_objects = []
                for i in range(len(scores)):
                    if scores[i] > self.min_conf_threshold:
                        ymin = int(max(1, (boxes[i][0] * self.DISPLAY_HEIGHT)))
                        xmin = int(max(1, (boxes[i][1] * self.DISPLAY_WIDTH)))
                        ymax = int(min(self.DISPLAY_HEIGHT, (boxes[i][2] * self.DISPLAY_HEIGHT)))
                        xmax = int(min(self.DISPLAY_WIDTH, (boxes[i][3] * self.DISPLAY_WIDTH)))
                        detected_objects.append([xmin, ymin, xmax, ymax])

                # Update tracking and draw interface
                frame = self._draw_interface(frame)
                self._update_tracking(detected_objects, frame)

                cv2.imshow('Advanced Object Detector', frame)

                if cv2.waitKey(1) == ord('q'):
                    break

        finally:
            self.video.release()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                       required=True)
    parser.add_argument('--graph', help='Name of the .tflite file',
                       default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file',
                       default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold',
                       default=0.5)
    parser.add_argument('--edgetpu', help='Use Edge TPU',
                       action='store_true')

    args = parser.parse_args()
    detector = ObjectDetector(args)
    detector.run()

if __name__ == '__main__':
    main()
