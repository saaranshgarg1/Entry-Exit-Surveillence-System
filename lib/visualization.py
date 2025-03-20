"""
Visualization utilities for depth maps and detection results
"""

import cv2
import numpy as np
from matplotlib import colormaps

def visualize_depth_with_matplotlib(depth2):
    """Visualize depth map with matplotlib colormap"""
    depth = depth2.copy()
    cmap = colormaps.get_cmap('Spectral_r')
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    
    # Display the depth map using OpenCV
    cv2.imshow("Depth Map with Spectral_r Colormap", depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
