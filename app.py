import sys
import cv2

from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtCore import  Qt
from PyQt5.QtMultimedia import  QMediaPlayer
from PyQt5.QtWidgets import (QApplication,  QHBoxLayout, QLabel, QMessageBox, QLineEdit,
        QPushButton, QStackedLayout,  QVBoxLayout, QWidget)

from PyQt5.QtCore import QMutex
import requests

from utilities.video import Video
from utilities.headpose_detector import HeadPoseDetector
from config.config import *
from utilities.thread_collections import SetPoseThread, FrameRenewerThread, PoseDetectorThread
from widgets.video_player import VideoPlayer


class Ui_MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.hp_detector = HeadPoseDetector()
        self.cam = Video()
        
        self.mut = QMutex()
        self.mut_file = QMutex()
        self.thread_set_pose = SetPoseThread(self.cam, self.mut, self.hp_detector)
        self.thread_renew_frame = FrameRenewerThread(self.cam, self.mut)

        
        self.set_ui()
        self.slot_init()
        self.setFocusPolicy(Qt.StrongFocus)
    
    
    def array_to_QPixmap(self, frame):    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        show_image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        show_image = QPixmap.fromImage(show_image)
        return show_image

    def set_ui(self):
        # wid = QtWidgets.QWidget(self)
        # self.setCentralWidget(wid)
        self.__layout_main = QHBoxLayout()  # 采用QHBoxLayout类，按照从左到右的顺序来添加控件
        self.__layout_fun_button = QVBoxLayout()
        self.__layout_frame_and_player = QStackedLayout()
        
        self.button_open_camera = QPushButton('Open Camera')
        self.button_close = QPushButton('Exit')
        self.label_text = QLabel()
        self.label_cam = QLabel()
        self.edit_user = QLineEdit()

        self.move(200, 200)

        # 信息显示
        self.label_show_camera = QLabel()
        
        # self.label_text.setFixedSize(200, 200)
        self.label_text.setText("Click Open Camera to Setup...")
        self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.setAutoFillBackground(False)
        

        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.edit_user)
        self.__layout_fun_button.addWidget(self.label_text)
        self.__layout_fun_button.addWidget(self.label_cam)        
        
        self.edit_user.setText("Default User")        
        
        
        self.__layout_frame_and_player.addWidget(self.label_show_camera)
        self.video_palyer = VideoPlayer()
        self.__layout_frame_and_player.addWidget(self.video_palyer)
        self.__layout_frame_and_player.setCurrentIndex(0)
        
        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addLayout(self.__layout_frame_and_player)

        self.label_cam.setVisible(False)

        self.setLayout(self.__layout_main)
        
        self.label_text.raise_()
        self.setWindowTitle('FocusApp')

    def renew_image(self, msg):
        show_image = self.array_to_QPixmap(msg["frame"])
        self.label_show_camera.setPixmap(show_image)
    
    def detect_pose(self, msg):
        if "error" in msg:
            return
        
        is_good = msg["is_good"]
        yaw = msg["yaw"]
        pitch = msg["pitch"]
        roll = msg["roll"]
        
        info = self.hp_detector.get_setting() + f"\n yaw = {yaw}\npitch={pitch}\nroll={roll}"

        if not is_good:
            if self.video_palyer.mediaPlayer.state() != QMediaPlayer.PlayingState:
                return
            self.video_palyer.pause_video()
            reply = QMessageBox.information(self, '', 'Please view the center of the screen!', QMessageBox.Yes)
            self.video_palyer.resume_video()
            print("Not Good")    
        
        # else: 
        #     # self.video_palyer.resume_video()
        #     print("Good")
        
        if not self.label_cam.isVisible():
            self.label_cam.setVisible(True)
        self.label_text.setText(info)
        show_image = self.array_to_QPixmap(msg["frame"])
        self.label_cam.setPixmap(show_image)
        
        
    def keyPressEvent(self, event):
        
        if event.key() == Qt.Key_Space:
            self.thread_set_pose.signal.connect(self.show_info)
            self.thread_set_pose.start()
        
            
    def show_info(self, msg):
        
        self.label_text.setText(msg["info"])
        if msg["status"] == DONE and not self.thread_renew_frame.isFinished():

            self.thread_renew_frame.release_cam()
            print("release cam")
            self.__layout_frame_and_player.setCurrentIndex(1)
            
            r = requests.post(POST_DATA_URL_USER, json=dict(name=self.edit_user.text()))
            self.edit_user.hide()
            self.thread_hp_detect = PoseDetectorThread(self.cam, self.mut, self.hp_detector, self.video_palyer, int(r.text))
            self.thread_hp_detect.signal.connect(self.detect_pose)
            self.thread_hp_detect.start()

            

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.button_close.clicked.connect(self.close)

    def button_open_camera_click(self):
        info = f"Please Look at Left most side of your screen.\n Press 'Space' to confirm \n 'c' to clear the selection."
        self.label_text.setText(info)
        self.thread_renew_frame.signal.connect(self.renew_image)
        if not self.thread_renew_frame.isRunning():
            self.thread_renew_frame.start()
            self.button_open_camera.setVisible(False)


    def closeEvent(self, event):
        ok = QPushButton()
        cancel = QPushButton()
        msg = QMessageBox(QMessageBox.Warning, 'Close', 'Want to Exit?')
        msg.addButton(ok, QMessageBox.ActionRole)
        msg.addButton(cancel, QMessageBox.RejectRole)
        ok.setText('Exit')
        cancel.setText('Cancel')
        if msg.exec_() == QMessageBox.RejectRole:
            event.ignore()
        else:
            event.accept()
            



if __name__ == '__main__':
    App = QApplication(sys.argv)
    win = Ui_MainWindow()
    win.show()
    sys.exit(App.exec_())
