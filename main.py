from config.config import *
from utilities.video import Video
from utilities.heatmap_generator import LTSLFM
from utilities.model import CNNModel
from utilities.util import visualize_result

import cv2
import numpy as np


if __name__ == '__main__':

    # create camera object
    camera = Video()
    print('Camera is set up...')
    
    model = CNNModel()
    print('Create model...')
    
    model.load(MODEL_PATH)
    print('Load Model Weights...')
    
    model.predict(np.zeros((136, 136, 3), 'float'))
    
    heatmap_generator = LTSLFM()
    
    heatmap = None
    frame_index = 0
    pred_exp = None
    
    while True:
        ret, frame = camera.get_frame()
        frame_index += 1
        if frame_index % SKIP_FRAMES != 0:
            continue
        if not ret:
            print('Can not open the specific camera or it is the last frame of the video.')
            break
        if frame is None:
            print('The video file is end!')
            break
        
        bbox, landmarks = heatmap_generator.extract_landmarks(frame)
        
        if len(heatmap_generator.landmarks_list) == 5:
            heatmap = heatmap_generator.construct_LTSLFM()
            pred_exp = model.predict(heatmap)
            
        visualize_result(frame, bbox, heatmap, landmarks, pred_exp, frame_index)
        cv2.imshow('out', frame)

        if cv2.waitKey(30) == 27:
            break
                
        
