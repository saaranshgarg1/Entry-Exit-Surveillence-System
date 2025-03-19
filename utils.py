"""
Utility functions for object detection and tracking
"""

import numpy as np

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
