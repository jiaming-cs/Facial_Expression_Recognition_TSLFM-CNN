import cv2
from config.config import CAMERA_TYPE
class Video:
    def __init__(self, source=None):
        if not source:
            source = CAMERA_TYPE
        self.__cam = cv2.VideoCapture(source)
        
    def open(self, source):
        self.__cam.open(source)
        
    def release(self):
        self.__cam.release()
        
    def is_open(self):
        return self.__cam.isOpened()
    
    def get_frame(self):
        ret, frame = self.__cam.read()
        return ret, frame  