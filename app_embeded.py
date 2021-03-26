import sys

import cv2
from PyQt5.QtGui import QIcon, QFont, QImage, QPixmap
from PyQt5.QtCore import QDir, Qt, QUrl, QSize
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
        QPushButton, QSizePolicy, QSlider, QStackedLayout, QStyle, QVBoxLayout, QWidget, QStatusBar)
from PyQt5.QtCore import QThread, QMutex
from PyQt5 import QtCore

from utilities.video import Video
from headpose_detector import HeadPoseDetector
from config.config import *

class VideoPlayer(QWidget):
    
    def __init__(self):
        super().__init__()

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        btnSize = QSize(16, 16)
        videoWidget = QVideoWidget()

        openButton = QPushButton("Open Video")   
        openButton.setToolTip("Open Video File")
        openButton.setStatusTip("Open Video File")
        openButton.setFixedHeight(24)
        openButton.setIconSize(btnSize)
        openButton.setFont(QFont("Noto Sans", 8))
        openButton.setIcon(QIcon.fromTheme("document-open", QIcon("D:/_Qt/img/open.png")))
        openButton.clicked.connect(self.abrir)

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setFixedHeight(24)
        self.playButton.setIconSize(btnSize)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.statusBar = QStatusBar()
        self.statusBar.setFont(QFont("Noto Sans", 7))
        self.statusBar.setFixedHeight(14)

        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(openButton)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        layout = QVBoxLayout()
        layout.addWidget(videoWidget)
        layout.addLayout(controlLayout)
        layout.addWidget(self.statusBar)

        self.setLayout(layout)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
        self.statusBar.showMessage("Ready")

    def abrir(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Selecciona los mediose",
                ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")

        if fileName != '':
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)
            self.statusBar.showMessage(fileName)
            self.play()
    def pause_video(self):
        self.mediaPlayer.pause()
    
    def resume_video(self):
        self.mediaPlayer.play()

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.statusBar.showMessage("Error: " + self.mediaPlayer.errorString())

class FrameRenewer(QThread):
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

            

            
class SetPose(QThread):
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
        
        
class PoseDetect(QThread):
    signal = QtCore.pyqtSignal(dict)  
    def __init__(self, cam, mut, hp_detector):
        super().__init__()
        self.cam = cam
        self.mut = mut
        self.hp_detector = hp_detector
        
    def run(self):
        while not self.cam.is_open():
            print("can't open")
            self.cam.open(0)

        while True:
            
            self.mut.lock()  
            ret, frame = self.cam.get_frame()
            self.mut.unlock()    
            if not ret:
                print("can not read from camera")
                break
            frame = cv2.resize(frame, (320, 240))
            try:
                is_good, yaw, pitch, roll = self.hp_detector.detect(frame)
                self.signal.emit(dict(is_good=is_good, yaw=yaw, pitch=pitch, roll=roll, frame=frame))
            except:
                self.signal.emit(dict(error=None))
            
    def release_cam(self):
        self.mut.lock()  
        self.cam.release()
        self.mut.unlock()
            
        

class Ui_MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.hp_detector = HeadPoseDetector()
        self.cam = Video()
        
        self.mut = QMutex()
        self.mut_file = QMutex()
        self.thread_set_pose = SetPose(self.cam, self.mut, self.hp_detector)
        self.thread_renew_frame = FrameRenewer(self.cam, self.mut)
        self.video_palyer = VideoPlayer()
        
        
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


        self.move(200, 200)

        # 信息显示
        self.label_show_camera = QLabel()
        
        # self.label_text.setFixedSize(200, 200)
        self.label_text.setText("Click Open Camera to Setup...")
        self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.setAutoFillBackground(False)
        
        
        

        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_close)

        self.__layout_fun_button.addWidget(self.label_text)
        self.__layout_fun_button.addWidget(self.label_cam)        
        
        
        self.__layout_frame_and_player.addWidget(self.label_show_camera)
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
            self.video_palyer.pause_video()
            print("Not Good")    
        else:
            self.video_palyer.resume_video()
            print("Good")
        
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
            self.thread_hp_detect = PoseDetect(self.cam, self.mut, self.hp_detector)
            self.thread_hp_detect.signal.connect(self.detect_pose)
            self.thread_hp_detect.start()
            
    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.button_close.clicked.connect(self.close)

        
    

    def button_open_camera_click(self):
        info = f"Please Look at Left most side of your screen.\n Press 'enter' to confirm \n 'c' to clear the selection."
        self.label_text.setText(info)
        self.thread_renew_frame.signal.connect(self.renew_image)
        print("is running?", self.thread_renew_frame.isRunning() )
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
            # if self.cap.isOpened():
            #     self.cap.release()
            # if self.timer_camera.isActive():
            #     self.timer_camera.stop()
            event.accept()
            



if __name__ == '__main__':
    App = QApplication(sys.argv)
    win = Ui_MainWindow()
    win.show()
    sys.exit(App.exec_())
