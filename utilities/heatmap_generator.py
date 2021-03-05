import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt


class LTSLFM:
    
    def __init__(self):    
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.draw_bbox = True
        self.draw_landmarks = True
        self.landmarks_list = []
        self.resize_factor = 0.5
        self.cmap = plt.get_cmap('jet')

    def __rect_to_bb(self, 
                     rect, 
                     resize_factor):
        x0 = int(rect.left() / resize_factor)
        y0 = int(rect.top() / resize_factor)
        x1 = int(rect.right() / resize_factor)
        y1 = int(rect.bottom() / resize_factor)
        return ((x0, y0), (x1, y1))

    def __shape_to_np(self, shape, resize_factor):
        coords = np.zeros((68, 2), dtype='int')
        for i in range(0, 68):
            coords[i] = (int(shape.part(i).x / resize_factor), int(shape.part(i).y / resize_factor))
        return coords

    def extract_landmarks(self, 
                          frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.resize(frame_gray, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        rects = self.detector(frame_gray, 1)
        shapes = []
        bbox = None
        for i, rect in enumerate(rects):
            shape = self.predictor(frame_gray, rect)
            shape = self.__shape_to_np(shape, self.resize_factor)
            bbox = self.__rect_to_bb(rect, self.resize_factor)
            shapes.append(shape)
        if len(shapes) == 0:
            landmarks = None
        else:
            landmarks = shapes[0]
            self.landmarks_list.append(landmarks)
        return bbox, landmarks
    
    def __normalize(self, landmarks):
        landmarks = landmarks.astype('float')
        l = np.linalg.norm(landmarks[27] - landmarks[8])
        for i in range(68):
            landmarks[i] = (landmarks[i] - landmarks[33]) / l
            # landmark[i] = (landmark[i] - landmark[33])
        
        return landmarks

    def construct_LTSLFM(self):
        normalized_landmarks = np.stack([self.__normalize(landmarks) for landmarks in self.landmarks_list])
        self.landmarks_list.clear()
        heat_maps = []
        for i in range(4):
            heat_map = np.zeros((68, 68))
            pre = normalized_landmarks[i]
            cur = normalized_landmarks[i+1]
            for i in range(68):
                for j in range(68):
                    heat_map[i, j] = np.abs((np.linalg.norm(pre[i] - pre[j]) - np.linalg.norm(cur[i] - cur [j])))
            heat_map /= np.amax(heat_map)
            heat_maps.append(heat_map)
        heat_maps = np.asarray(heat_maps)
        concatenated_heatmap = []
        for i in range(2):
            row = []
            for j in range(2):
                row.append(heat_maps[i*2+j])
            concatenated_heatmap.append(np.hstack(row))
        concatenated_heatmap = np.vstack(concatenated_heatmap)
        rgb_heatmap = np.delete(self.cmap(concatenated_heatmap), 3, 2)
        return rgb_heatmap


    