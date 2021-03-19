import cv2
from config.config import *

def draw_lines(img, position, lines):
    for i, line in enumerate(lines):
        p = (position[0], position[1] - (len(lines) - i - 1) * LINE_HEIGHT)
        cv2.putText(img, line, p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    


def visualize_result(frame,
                    bbox = None,
                    heatmap = None,
                    landmarks = None, 
                    pred_exp = None, 
                    frame_index = None,
                    ):
    if bbox is not None:
        p0, p1 = bbox
        cv2.rectangle(frame, p0, p1, BBOX_COLOR, BBOX_THINKNESS)
    else:
        p = (0, 50)
        cv2.putText(frame, "No Face Detected in Current Frame", p, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THINKNESS)
        return     
    
    if heatmap is not None:
        h, w, _ = heatmap.shape
        heatmap = heatmap * 255
        heatmap = heatmap.astype('uint8')
        frame[-h:, -w:] = heatmap
        
    if landmarks is not None:
        for p in landmarks:
            frame = cv2.circle(frame, (p[0], p[1]), LANDMARKS_RADIOUS, LANDMARKS_COLOR, -1)
    
    if pred_exp is not None:
        p0, p1 = bbox    
        cv2.putText(frame, expression_mapping[pred_exp], p0, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THINKNESS)
        
    if frame_index is not None:
        p = (0, frame.shape[0])
        cv2.putText(frame, f'Current Frame: {frame_index}', p, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_COLOR, FONT_THINKNESS)
    