import numpy as np
import cv2
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class TrackingConfig:
    """Configuration parameters for object tracking"""
    initial_window: Tuple[int, int, int, int]  # (x, y, width, height)
    hsv_min: Tuple[float, float, float] = (0., 60., 32.)
    hsv_max: Tuple[float, float, float] = (180., 255., 255.)
    term_criteria_max_iter: int = 10
    term_criteria_epsilon: float = 1.0

class ObjectTracker:
    """Class for object tracking using CAMShift algorithm"""
    
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.track_window = config.initial_window
        self.term_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            config.term_criteria_max_iter,
            config.term_criteria_epsilon
        )
        self.roi_hist = None
        self.tracking_started = False
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.selection = None

    def mouse_drawing(self, event, x, y, flags, params):
        """Handle mouse events for ROI selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.selection = None
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.selection = (min(self.ix, x), min(self.iy, y), 
                              abs(x - self.ix), abs(y - self.iy))
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if x != self.ix and y != self.iy:
                self.selection = (min(self.ix, x), min(self.iy, y), 
                                  abs(x - self.ix), abs(y - self.iy))
                self.track_window = self.selection
                self.tracking_started = True

    def initialize_roi(self, frame: np.ndarray) -> bool:
        """Initialize Region of Interest (ROI) for tracking"""
        try:
            if self.track_window and all(v > 0 for v in self.track_window):
                x, y, w, h = self.track_window
                roi = frame[y:y+h, x:x+w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                mask = cv2.inRange(
                    hsv_roi,
                    np.array(self.config.hsv_min),
                    np.array(self.config.hsv_max)
                )
                
                self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
                return True
            return False
        except Exception as e:
            print(f"Error initializing ROI: {e}")
            return False

    def track(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Track object in the current frame"""
        try:
            if self.roi_hist is not None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
                
                ret, self.track_window = cv2.CamShift(dst, self.track_window, self.term_criteria)
                
                pts = cv2.boxPoints(ret)
                return np.int32(pts)
            return None
        except Exception as e:
            print(f"Error during tracking: {e}")
            return None

def process_camera(config: TrackingConfig, camera_id: int = 0):
    """Process webcam feed with object tracking"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Could not open camera {camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = ObjectTracker(config)
    window_name = 'Object Tracking'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, tracker.mouse_drawing)

    print("Instructions:")
    print("1. Click and drag to select tracking region")
    print("2. Press 'q' to quit")
    print("3. Press 'r' to reset tracking")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_display = frame.copy()

            if tracker.selection is not None:
                x, y, w, h = tracker.selection
                cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if tracker.tracking_started and tracker.roi_hist is None:
                if tracker.initialize_roi(frame):
                    print("Tracking initialized!")
                else:
                    tracker.tracking_started = False
                    print("Failed to initialize tracking. Please try again.")

            if tracker.tracking_started and tracker.roi_hist is not None:
                pts = tracker.track(frame)
                if pts is not None:
                    cv2.polylines(frame_display, [pts], True, (0, 255, 0), 2)
                    x, y, w, h = tracker.track_window
                    cv2.putText(frame_display, f"Position: ({x}, {y})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, frame_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracker.tracking_started = False
                tracker.roi_hist = None
                tracker.selection = None
                print("Tracking reset. Select new region.")

    except Exception as e:
        print(f"Error during video processing: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Object Tracking using CAMShift')
    parser.add_argument("-c", "--camera", type=int, default=0, help='Camera device index (default: 0)')
    parser.add_argument("-x", "--x", type=int, default=300, help='Initial window X position')
    parser.add_argument("-y", "--y", type=int, default=200, help='Initial window Y position')
    parser.add_argument("-w", "--width", type=int, default=100, help='Initial window width')
    parser.add_argument("--height", type=int, default=50, help='Initial window height')

    args = parser.parse_args()

    config = TrackingConfig(
        initial_window=(args.x, args.y, args.width, args.height)
    )

    process_camera(config, args.camera)

if __name__ == "__main__":
    main()
