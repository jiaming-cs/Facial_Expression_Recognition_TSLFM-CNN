import cv2
from config.config import CAMERA_TYPE
class Camera:
    def __init__(self):
        self.__cam = cv2.VideoCapture(CAMERA_TYPE)
        
    def get_frame(self):
        ret, frame = self.__cam.read()
        return ret, frame  