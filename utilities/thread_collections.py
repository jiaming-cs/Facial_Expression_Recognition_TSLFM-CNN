from threading import Thread
from PyQt5 import QtCore
from PyQt5.QtCore import QThread
import cv2

import datetime

class FrameRenewerThread(QThread):
    signal = QtCore.pyqtSignal(dict)    
    
    def __init__(self, cam, mut):
        super().__init__()
        self.cam = cam
        self.mut = mut  
          
    def run(self):  
        while True:
            self.mut.lock()  
            ret, frame = self.cam.get_frame()
            self.mut.unlock()    
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            self.signal.emit(dict(frame = frame))
            
    def release_cam(self):
        self.mut.lock()  
        self.cam.release()
        self.mut.unlock()
        

 
class SetPoseThread(QThread):
    signal = QtCore.pyqtSignal(dict)  
    def __init__(self, cam, mut, hp_detector):
        super().__init__()
        self.cam = cam
        self.mut = mut
        self.hp_detector = hp_detector
        
    def run(self):
        self.mut.lock()   
        ret, frame = self.cam.get_frame()
        self.mut.unlock()
        ret_status, info = self.hp_detector.setup(frame)
        self.signal.emit(dict(status = ret_status, info = info))
        

        
              
class PoseDetectorThread(QThread):
    signal = QtCore.pyqtSignal(dict)  
    def __init__(self, cam, mut, hp_detector, video_player):
        super().__init__()
        self.cam = cam
        self.mut = mut
        self.hp_detector = hp_detector
        self.video_palyer = video_player
        
    def run(self):
        while not self.cam.is_open():
            print("can't open")
            self.cam.open(0)
            
        frame_index = 0
        while not self.video_palyer.video_name:
            QThread.sleep(1)
            print("sleep...")
        
        video_name = self.video_palyer.video_name
        while True:
            self.mut.lock()  
            ret, frame = self.cam.get_frame()
            self.mut.unlock()    
            if not ret:
                print("can not read from camera")
                break
            frame = cv2.resize(frame, (320, 240))
            # try:
                
            is_good, yaw, pitch, roll = self.hp_detector.detect(frame, frame_index, datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), video_name) 

            self.signal.emit(dict(is_good=is_good, yaw=yaw, pitch=pitch, roll=roll, frame=frame))
            
            # except:
            #     self.signal.emit(dict(error=None))
            
    def release_cam(self):
        self.mut.lock()  
        self.cam.release()
        self.mut.unlock()