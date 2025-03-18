"""
Real-time Person Tracking System with Door Entry/Exit Detection
Features:
- Person detection and tracking using YOLO
- Face recognition with InsightFace
- Depth-based door detection
- Entry/exit logging with confidence scores
"""

import cv2
import time
import torch
from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO
import numpy as np
import os
from queue import Queue
from threading import Thread
import insightface
from scipy.spatial.distance import cosine
from matplotlib import colormaps
import csv
from datetime import datetime

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

def associate_face_person(persons, faces, face_idx, associations):
    """Associate faces with persons based on overlap"""
    current_face = faces[face_idx]
    face_box = current_face  # Ensure this is actually a bounding box [x1, y1, x2, y2]
    
    max_face_overlap = 0.0
    max_body_overlap = 0.0
    best_person_idx = None
    
    # Find best matching person
    for person_idx in range(len(persons)):
        # Skip if person already matched
        if not associations[person_idx][3]:
            continue
        
        person_box = persons[person_idx].xyxy[0]
        
        # Calculate intersection
        x1 = max(person_box[0], face_box[0])
        y1 = max(person_box[1], face_box[1])
        x2 = min(person_box[2], face_box[2])
        y2 = min(person_box[3], face_box[3])
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        intersection = (x2 - x1) * (y2 - y1)
        face_area = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])
        body_area = (person_box[2] - person_box[0]) * (person_box[3] - person_box[1])
        
        # Avoid division by zero
        if face_area <= 0 or body_area <= 0:
            continue
        
        face_overlap = intersection / face_area
        body_overlap = intersection / body_area
        
        # Update best match
        if face_overlap > max_face_overlap:
            max_face_overlap = face_overlap
            max_body_overlap = body_overlap
            best_person_idx = person_idx
        elif face_overlap == max_face_overlap and body_overlap > max_body_overlap:
            max_body_overlap = body_overlap
            best_person_idx = person_idx
    
    # No overlap found
    if max_face_overlap == 0.0:
        return associations
    
    new_associations = associations.copy()
    
    # If better match found, update associations
    if best_person_idx is not None:
        curr_overlap = new_associations[best_person_idx][2]
        if curr_overlap < max_body_overlap:
            old_face = new_associations[best_person_idx][1]
            if old_face != 0 and old_face in faces:
                old_face_idx = faces.index(old_face)
                # Only re-associate if old_face_idx is valid
                if 0 <= old_face_idx < len(faces):
                    new_associations = associate_face_person(persons, faces, old_face_idx, new_associations)
            
            # Update with new face
            new_associations[best_person_idx] = [
                persons[best_person_idx],   # Person bounding box
                current_face,               # Person face
                max_body_overlap,           # Overlap
                1,                          # Available
                "Unknown",                  # Identity
                1,                          # Min distance
                1                           # Inside or outside
            ]
    
    return new_associations

def load_faces(model):
    faces_dir = "faces"
    database = {}
    for filename in os.listdir(faces_dir):
        if filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".jpg"):
            filepath = os.path.join(faces_dir, filename)
            img = cv2.imread(filepath)

            # Detect and extract face embeddings
            faces = model.get(img, max_num=1)
            if faces:
                embedding = faces[0].embedding  # Assume one face per image
                name = os.path.splitext(filename)[0]  # e.g., 'abc' from 'abc.png'
                database[name] = embedding
    return database

def match_face(face_data, database):
    """Match detected face with database"""
    if face_data is None or len(face_data) == 0:
        return "Unknown", 1.0
        
    min_dist = float('inf')
    identity = "Unknown"
    
    if not database:
        return identity, min_dist
        
    try:
        for name, db_embedding in database.items():
            dist = cosine(face_data['embedding'], db_embedding)
            if dist < min_dist:
                min_dist = dist
                identity = name
        if min_dist > 0.7:
            identity = "Unknown"
    except Exception as e:
        print(f"Error matching face: {e}")
        return "Unknown", 1.0
        
    return identity, min_dist

def calculate_IOU(boxA, boxB):
    """Calculate Intersection over Union (IoU)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxB[3], boxB[3])
    
    # Calculate intersection area
    intersection = max(0, xB - xA) * max(0, yB - yA)
    
    # Calculate area of both boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Calculate union area
    union = boxA_area + boxB_area - intersection
    
    # Calculate IoU
    iou = intersection / union
    return iou

def IOU_tracker(persons, prev_persons):
    """Track persons using Intersection over Union (IoU)"""
    if not prev_persons:
        return persons
    if len(persons) != len(prev_persons):
        return -1
        
    new_persons = {}
    for key,value in prev_persons.items():
        max_iou = 0
        best_match = None
        for key2,value2 in persons.items():
            iou = calculate_IOU(value[0].xyxy[0], value2[0].xyxy[0])
            if iou > max_iou:
                max_iou = iou
                best_match = value2
        if max_iou > 0.5:
            if type(best_match[1])!=int and type(value[1])==int:
                return -1
            if type(best_match[1])==int and type(value[1])!=int:
                new_persons[key] = [best_match[0], value[1], best_match[2], best_match[3], value[4], value[5], value[6]]
            else:
                new_persons[key] = [best_match[0], best_match[1], best_match[2], best_match[3], value[4], value[5], value[6]]
        else:
            return -1
    return new_persons
    
def visualize_depth_with_matplotlib(depth2):
    depth = depth2.copy()
    cmap = colormaps.get_cmap('Spectral_r')
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    
    # Display the depth map using OpenCV
    cv2.imshow("Depth Map with Spectral_r Colormap", depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def detect_door_from_depth(depth_map):
    # Normalize the depth map
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # depth_normalized = depth_map.copy().astype(np.uint8)
    
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
        
        print(len(approx), len(approx2))
        print(cv2.boundingRect(approx)[2]*cv2.boundingRect(approx)[3], cv2.boundingRect(approx2)[2]*cv2.boundingRect(approx2)[3])
    
    try:
        print(cv2.boundingRect(door_contour)[2]*cv2.boundingRect(door_contour)[3])
    except:
        pass
    
    cv2.imshow("All Contours", depth_normalized)
    if door_contour is not None:
        cv2.drawContours(result, [door_contour], -1, (0, 255, 0), 3)  # Green contour
    
    cv2.imshow("Door Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Return the bounding box of the detected door after converting to x1, y1, x2, y2 format and adding a margin that is 10% of the width and height and is inside the image
    if door_contour is not None:
        x, y, w, h = cv2.boundingRect(door_contour)
        margin_w = int(w * 0.3)
        margin_h = int(h * 0.3)
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(depth_normalized.shape[1], x + w + margin_w)
        y2 = min(depth_normalized.shape[0], y + h + margin_h)
        print(x1, y1, x2, y2)
        return (x1, y1, x2, y2)
    else:
        return None


def detect_door(frame, model):
    perf_counter = time.perf_counter()
    depth_map = model.infer_image(frame)
    np.savetxt("depth2.csv", depth_map, delimiter=",")
    print(depth_map) # Prints max element of the depth map
    print(np.max(depth_map))
    print(np.min(depth_map))
    print(f"Time taken for depth map: {time.perf_counter() - perf_counter:.5f}")
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

class PersonTracker:
    def __init__(self):
        self.person_states = {}  # {person_id: [identity, last_state, last_seen, confidence]}
        self.log_file = 'person_tracking.csv'
        self.initialize_log()
        
    def initialize_log(self):
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Person', 'Action', 'Confidence'])
    
    def update_state(self, person_id, identity, current_state, confidence):
        timestamp = datetime.now()
        
        # Get previous state
        prev_state = None
        if person_id in self.person_states:
            prev_state = self.person_states[person_id][1]
        
        # Update state if changed
        if prev_state != current_state:
            action = "Entered" if current_state == 0 else "Exited"
            self.log_activity(timestamp, identity, action, confidence)
        
        # Update tracking
        self.person_states[person_id] = [identity, current_state, timestamp, confidence]
    
    def log_activity(self, timestamp, identity, action, confidence):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, identity, action, confidence])
    
    def handle_disappearance(self, current_persons):
        """Handle persons who disappeared from frame"""
        timestamp = datetime.now()
        disappeared = set(self.person_states.keys()) - set(current_persons)
        
        for person_id in disappeared:
            if (timestamp - self.person_states[person_id][2]).seconds > 5:
                # Person truly disappeared, log final state
                identity = self.person_states[person_id][0]
                last_state = "Inside" if self.person_states[person_id][1] == 0 else "Outside"
                self.log_activity(timestamp, identity, f"Lost tracking ({last_state})", 
                                self.person_states[person_id][3])
                del self.person_states[person_id]

def webcam_feed():
    # init webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # YOLO load
    model = YOLO('yolo11n.pt', verbose=False).to(device)
    # face_model = YOLO('yolov8n-face-lindevs.pt').to('cuda')
    
    # Depth Load
    model_depth = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]).to(device).eval()
    model_depth.load_state_dict(torch.load('depth_anything_v2_vitl.pth', map_location=device))
    frame = cap.read()[1]
    door_bbox = detect_door(frame, model_depth) # Door detection
    
    # Depth Load 2
    model_depth2 = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384]).to(device).eval()
    model_depth2.load_state_dict(torch.load('depth_anything_v2_vits.pth', map_location=device))
    depth = model_depth2.infer_image(frame[door_bbox[1]:door_bbox[3], door_bbox[0]:door_bbox[2]])
    avg_depth_door = np.mean(depth)
    
    # Load faces
    model_recog = insightface.app.FaceAnalysis(name="buffalo_sc", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    model_recog.prepare(ctx_id=0) 
    database = load_faces(model_recog)
    print(f"Loaded {len(database)} faces {database.keys()}")
    
    # Previous values
    prev_frame_time = time.perf_counter()
    prev_persons = {}
    prev_faces = []
    prev_no_persons = 0
    prev_no_faces = 0
    frames = 0
    door_activity_frames = 0
    cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Object Detection', 1280, 720)
    
    # Initialize person tracker
    person_tracker = PersonTracker()
    
    try:
        while True:
            # Extract frame
            ret, frame = cap.read()
            frames += 1
            if not ret:
                print("Failed to grab frame")
                break
                
            # Run YOLO detection
            # t1 = time.perf_counter()
            results = model(frame, verbose=False)
            # print(f"Time taken for YOLO: {time.perf_counter() - t1:.5f}")
            annotated_frame = frame.copy()
            persons = [a for a in results[0].boxes if a.cls == 0]
            for person in persons:
                b = person.xyxy[0]
                annotated_frame = cv2.rectangle(annotated_frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
            no_persons = len(persons)
            
            # USING INSIGHTFACE INSTEAD OF YOLOFACE
            faces, kpss = model_recog.det_model.detect(frame, metric='default')
            for bbox in faces:
                            annotated_frame = cv2.rectangle(annotated_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                        
            # Associate faces with persons
            t1 = time.perf_counter()
            dict_person_face = {i: [persons[i],0,0,1,"Unknown",1,1] for i in range(len(persons))}
            for i in range(len(faces)):
                dict_person_face = associate_face_person(persons,faces,i,dict_person_face)     
                for key,value in dict_person_face.items():
                    if value[3] == 0:
                        dict_person_face[key][3] = 1   
                        
            # Draw associations
            for key,value in dict_person_face.items():
                if type(value[1]) != int:
                    person_box = value[0].xyxy[0]
                    face_box = value[1]
                    cv2.line(annotated_frame, (int((person_box[0]+person_box[2])/2), int((person_box[1]+person_box[3])/2)), 
                             (int((face_box[0]+face_box[2])/2), int((face_box[1]+face_box[3])/2)), (0, 255, 0), 2)  
            
            IOU = IOU_tracker(dict_person_face, prev_persons)  
            if IOU !=-1:
                no_faces = sum(1 for v in IOU.values() if type(v[1]) != int)
            if no_persons != prev_no_persons or no_faces != prev_no_faces or frames>=50 or IOU == -1:
                faces = model_recog.get2(frame, faces, kpss)
                facesboxes = [a['bbox'] for a in faces]

                for key,value in dict_person_face.items():
                    if type(value[1]) != int:
                        face_data = dict(faces[next((i for i, box in enumerate(facesboxes) if np.array_equal(np.array(box), np.array(value[1][:4]))), None)])
                        # priqnt(face_data)
                        identity, min_dist = match_face(face_data, database)
                        dict_person_face[key] = [value[0], value[1], value[2], value[3], identity, min_dist, value[6]]
                        cv2.putText(annotated_frame, f"{identity} ({min_dist:.2f}) : {"Inside" if value[6] else "Outside"}", 
                                (int(value[0].xyxy[0][0]), int(value[0].xyxy[0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0) if value[6] else (0,0,255), 2)
                    else:
                        dict_person_face[key] = [value[0], value[1], value[2], value[3], "Unknown", 1, value[6]]
                        cv2.putText(annotated_frame, f"Unknown :  : {"Inside" if value[6] else "Outside"}", 
                                (int(value[0].xyxy[0][0]), int(value[0].xyxy[0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0) if value[6] else (0,0,255), 2)
                # print("\r\r\rmode 1")
                frames = 0
                
            else:
                dict_person_face = IOU
                for key,value in dict_person_face.items():
                    cv2.putText(annotated_frame, f"{value[4]} ({value[5]:.2f}) : {"Inside" if value[6] else "Outside"}", 
                                (int(value[0].xyxy[0][0]), int(value[0].xyxy[0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0) if value[6] else (0,0,255), 2)
                # print(dict_person_face)
                # print("\r\rmode 2")
            
            # FPS
            current_time = time.perf_counter()
            fps = 1 / (current_time - prev_frame_time)
            prev_frame_time = current_time
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2)
            
            ## Door activity detection
            if door_bbox is not None:
                flag = 1
                list_to_edit = []
                x1, y1, x2, y2 = door_bbox
                i=0
                for person in persons:
                    iou = calculate_IOU(person.xyxy[0], door_bbox[:4])
                    i+=1
                    if iou > 0.3 or (person.xyxy[0][0] > x1 and person.xyxy[0][1] > y1 and person.xyxy[0][2] < x2 and person.xyxy[0][3] < y2):
                        flag = 0
                        if door_activity_frames == 2:
                            list_to_edit.append(i)
                        else:    
                            break
                
                if not flag:
                    if door_activity_frames == 2:
                        depth_map = model_depth2.infer_image(frame[door_bbox[1]:door_bbox[3], door_bbox[0]:door_bbox[2]])
                        
                        # Get person boxes for visualization
                        person_boxes = [persons[i-1].xyxy[0] for i in list_to_edit]
                        
                        # Create and show heatmap
                        heatmap = visualize_depth_heatmap(depth_map, person_boxes, door_bbox)
                        cv2.imshow("Depth Heatmap", heatmap)
                        
                        for i in list_to_edit:
                            person = persons[i-1]
                            person_box = person.xyxy[0]
                            x1 = int(max(0, person_box[0]-door_bbox[0]))
                            y1 = int(max(0, person_box[1]-door_bbox[1]))
                            x2 = int(min(door_bbox[2]-door_bbox[0], person_box[2]-door_bbox[0]))
                            y2 = int(min(door_bbox[3]-door_bbox[1], person_box[3]-door_bbox[1]))
                            avg_depth = np.mean(depth_map[y1:y2, x1:x2])
                            print(avg_depth)
                            avg_depth_door = np.mean(depth_map)
                            print(avg_depth_door)
                            if avg_depth < avg_depth_door:
                                dict_person_face[i-1][6] = 0
                            else:
                                dict_person_face[i-1][6] = 1
                                
                            # Update person state
                            current_state = dict_person_face[i-1][6]
                            identity = dict_person_face[i-1][4]
                            confidence = 1.0 / dict_person_face[i-1][5] if dict_person_face[i-1][5] != 0 else 0
                            
                            person_tracker.update_state(i-1, identity, current_state, confidence)
                        
                        # Handle disappearances
                        current_person_ids = [i-1 for i in list_to_edit]
                        person_tracker.handle_disappearance(current_person_ids)
                        
                        door_activity_frames = 0
                    door_activity_frames += 1
                        
            
            # Display frame
            cv2.imshow('Object Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            prev_persons = dict_person_face
            prev_faces = faces
            prev_no_persons = no_persons
            prev_no_faces = no_faces
            
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_feed()
