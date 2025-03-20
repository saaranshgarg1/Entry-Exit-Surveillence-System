"""
Depth processing utilities for door detection and depth analysis
"""

import cv2
import time
import torch
import numpy as np
from queue import Queue
from threading import Thread
from depth_anything_v2.dpt import DepthAnythingV2
from matplotlib import colormaps

class AsyncDepthProcessor:
    def __init__(self, model):
        self.model = model
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.running = True
        self.initialized = False
        self.error = None
        # Start worker thread with error handling
        self.worker = Thread(target=self._safe_process_frames)
        self.worker.daemon = True  # Allow cleanup on program exit
        self.worker.start()
        self.currently_processing = False
        
        # Wait for initialization with timeout
        init_timeout = 5  # seconds
        start_time = time.time()
        while not self.initialized and not self.error:
            if time.time() - start_time > init_timeout:
                self.stop()
                raise TimeoutError("Depth processor initialization timed out")
            time.sleep(0.1)
            
        if self.error:
            raise RuntimeError(f"Depth processor initialization failed: {self.error}")

    def _safe_process_frames(self):
        """Worker thread with error handling"""
        try:
            # Verify CUDA availability for thread
            if torch.cuda.is_available():
                torch.cuda.set_device(torch.cuda.current_device())
            
            # Signal successful initialization
            self.initialized = True
            
            while self.running:
                try:
                    if not self.frame_queue.empty():
                        frame = self.frame_queue.get_nowait()
                        with torch.no_grad():
                            depth_map = self.model.infer_image(frame)
                            self.result_queue.put_nowait(depth_map)
                            self.currently_processing = False
                    else:
                        time.sleep(0.01)  # Prevent busy waiting
                except Queue.Empty:
                    continue
                
        except Exception as e:
            self.error = str(e)
            self.running = False

    def process_frame(self, frame):
        if not self.running:
            raise RuntimeError("Depth processor stopped due to error")
        self.currently_processing = True
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except Queue.Empty:
                pass
        self.frame_queue.put_nowait(frame)

    def get_result(self):
        try:
            return self.result_queue.get_nowait() if not self.result_queue.empty() else None
        except Queue.Empty:
            return None
    
    def queue_length(self):
        return self.frame_queue.qsize() + (1 if self.currently_processing else 0)
    
    def stop(self):
        self.running = False
        if self.worker.is_alive():
            self.worker.join(timeout=1.0)

def detect_door_from_depth(depth_map):
    # Normalize the depth map
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply Gaussian blur to reduce noise
    depth_normalized = cv2.GaussianBlur(depth_normalized, (5, 5), 0)
    
    for i in range(len(depth_normalized)):
        for j in range(len(depth_normalized[0])):
            if depth_normalized[i][j] > 50:
                depth_normalized[i][j] = 50
    
    # Edge detection
    edges = cv2.Canny(depth_normalized, 5, 50, L2gradient=True)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours for quadrilateral shapes
    door_contour = None
    max_area = 4000
    result = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, contours, -1, (0, 0, 255), 3)
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.03*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx2 = cv2.approxPolyDP(contour, epsilon, False)
        
        # Check if it's a quadrilateral
        if len(approx) >=4:
            w,h = cv2.minAreaRect(approx)[1]
            if not (h/w < 0.2 or h/w > 5):
                area = w * h
                if area > max_area:  # Find the largest quadrilateral
                    max_area = area
                    door_contour = approx
        elif len(approx2) >=3: # Check if it's a Open Quadrilateral whose one side wasnt detected
            w,h = cv2.minAreaRect(approx)[1]
            if not (h/w < 0.2 or h/w > 5):
                area = w * h
                if area > max_area:
                    max_area = area
                    door_contour = approx2
        
        if len(approx) >=3:
            result = cv2.drawContours(result, [approx], -1, (255, 0, 0), 3)
            bound = cv2.boundingRect(door_contour)
            result = cv2.putText(result, f"{len(approx)}", (bound[0], bound[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            result = cv2.drawContours(result, [approx2], -1, (0, 0, 255), 3)
            bound = cv2.boundingRect(approx2)
            result = cv2.putText(result, f"{ len(approx2) }", (bound[0], bound[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # print(len(approx), len(approx2))
        # print(cv2.boundingRect(approx)[2]*cv2.boundingRect(approx)[3], cv2.boundingRect(approx2)[2]*cv2.boundingRect(approx2)[3])
    
    # try:
    #     print(cv2.boundingRect(door_contour)[2]*cv2.boundingRect(door_contour)[3])
    # except:
    #     pass
    
    cv2.imshow("All Contours", depth_normalized)
    if door_contour is not None:
        cv2.drawContours(result, [door_contour], -1, (0, 255, 0), 3)  # Green contour
    
    cv2.imshow("Door Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Return the bounding box of the detected door
    if door_contour is not None:
        x, y, w, h = cv2.boundingRect(door_contour)
        margin_w = int(w * 0.3)
        margin_h = int(h * 0.3)
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(depth_normalized.shape[1], x + w + margin_w)
        y2 = min(depth_normalized.shape[0], y + h + margin_h)
        # print(x1, y1, x2, y2)
        return (x1, y1, x2, y2)
    else:
        return None

def detect_door(frame, model):
    perf_counter = time.perf_counter()
    depth_map = model.infer_image(frame)
    np.savetxt("depth2.csv", depth_map, delimiter=",")
    # print(depth_map) # Prints max element of the depth map
    # print(np.max(depth_map))
    # print(np.min(depth_map))
    print(f"Time taken for depth map: {time.perf_counter() - perf_counter:.5f}")
    
    # Import here to avoid circular imports
    from .visualization import visualize_depth_with_matplotlib
    visualize_depth_with_matplotlib(depth_map)
    
    door_candidates = detect_door_from_depth(depth_map)
    return door_candidates

def visualize_depth_heatmap(depth_map, person_boxes, door_bbox):
    """Create colored heatmap with person boxes"""
    # Normalize depth values
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    
    # Draw person boxes
    for box in person_boxes:
        x1 = int(max(0, box[0]-door_bbox[0]))
        y1 = int(max(0, box[1]-door_bbox[1]))
        x2 = int(min(door_bbox[2]-door_bbox[0], box[2]-door_bbox[0]))
        y2 = int(min(door_bbox[3]-door_bbox[1], box[3]-door_bbox[1]))
        
        # Draw rectangle
        cv2.rectangle(heatmap, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add depth value
        avg_depth = np.mean(depth_map[y1:y2, x1:x2])
        cv2.putText(heatmap, f"d:{avg_depth:.2f}", 
                   (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
    
    return heatmap
