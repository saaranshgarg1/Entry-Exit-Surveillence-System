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
import numpy as np
from ultralytics import YOLO

# Import our modules
from depth_processor import DepthAnythingV2, detect_door, visualize_depth_heatmap
from person_tracker import PersonTracker
from face_recognition import load_faces, match_face, associate_face_person
from utils import calculate_IOU, IOU_tracker
from face_analysis_wrapper import EnhancedFaceAnalysis

def webcam_feed():
    # init webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # YOLO load
    model = YOLO('yolo11n.pt', verbose=False).to(device)
    
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
    model_recog = EnhancedFaceAnalysis(name="buffalo_sc", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
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
            results = model(frame, verbose=False)
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
                        identity, min_dist = match_face(face_data, database)
                        dict_person_face[key] = [value[0], value[1], value[2], value[3], identity, min_dist, value[6]]
                        cv2.putText(annotated_frame, f"{identity} ({min_dist:.2f}) : {'Inside' if value[6] else 'Outside'}", 
                                (int(value[0].xyxy[0][0]), int(value[0].xyxy[0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0) if value[6] else (0,0,255), 2)
                    else:
                        dict_person_face[key] = [value[0], value[1], value[2], value[3], "Unknown", 1, value[6]]
                        cv2.putText(annotated_frame, f"Unknown :  : {'Inside' if value[6] else 'Outside'}", 
                                (int(value[0].xyxy[0][0]), int(value[0].xyxy[0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0) if value[6] else (0,0,255), 2)
                frames = 0
                
            else:
                dict_person_face = IOU
                for key,value in dict_person_face.items():
                    cv2.putText(annotated_frame, f"{value[4]} ({value[5]:.2f}) : {'Inside' if value[6] else 'Outside'}", 
                                (int(value[0].xyxy[0][0]), int(value[0].xyxy[0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0) if value[6] else (0,0,255), 2)
            
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
                            avg_depth_door = np.mean(depth_map)
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
