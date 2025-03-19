"""
Face recognition and face-person association functions
"""

import os
import cv2
import numpy as np
from scipy.spatial.distance import cosine

def load_faces(model):
    """Load face embeddings from the faces directory"""
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
