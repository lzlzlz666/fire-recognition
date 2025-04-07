import cv2
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from Vedio import Ui_Form  # 导入子 UI 界面
from ultralytics import YOLO

# # 默认 YOLO 模型路径
# MODEL_PATH = r"D:\python_projects\fire-recognition\weights\lz-train-base-yolo11n.pt"
# # 通用 YOLO 模型路径
# MODEL_PATH_YOLO = r"D:\python_projects\fire-recognition\weights\yolo11n.pt"
MODEL_PATH_FILE = os.path.join("weights", "lz-train-base-yolo11n.pt")
MODEL_PATH_YOLO = os.path.join("weights", "yolo11n.pt")

class secondwindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(secondwindow, self).__init__()
        self.setupUi(self)
        self.setWindowFlags(Qt.Window)
        self.video_source = None
        self.cap = None
        self.model = None  # 初始化为空，点击按钮时再加载模型

        self.camera_2.setAutoExclusive(False)

        # 绑定按钮事件
        self.video_input_2.clicked.connect(self.open_video)
        self.camera_2.clicked.connect(self.toggle_camera)
        self.det.clicked.connect(lambda: self.start_detection_button(MODEL_PATH_FILE))         # 默认模型
        self.det_3.clicked.connect(lambda: self.start_detection_button(MODEL_PATH_YOLO))  # 通用模型
        self.det_2.clicked.connect(self.stop_detection_button)

    def start_detection_button(self, model_path):
        """ 根据传入的模型路径进行检测 """
        if self.video_source is None:
            QMessageBox.warning(self, "警告", "请先选择视频文件或摄像头！")
            return

        if self.cap and self.cap.isOpened():
            QMessageBox.warning(self, "警告", "检测已经在进行中，请先停止当前检测")
            return

        self.model = YOLO(model_path)  # 动态加载模型
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开视频源！")
            return

        self.process_video()

    def stop_detection_button(self):
        """ 停止检测，并重置 """
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.frame_label_2.clear()
            QMessageBox.information(self, "提示", "检测已停止")
            self.model = None  # 可选：停止后清空模型
        else:
            QMessageBox.warning(self, "警告", "当前没有正在进行的检测")

    def process_video(self):
        """ 视频帧处理逻辑 """
        while self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model(frame)
            for result in results:
                frame = result.plot()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            qimg = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)

            # 按比例缩放到标签大小
            pixmap = QPixmap.fromImage(qimg)
            scaled_pixmap = pixmap.scaled(self.frame_label_2.width(), self.frame_label_2.height(), Qt.KeepAspectRatio)

            self.frame_label_2.setPixmap(scaled_pixmap)
            self.frame_label_2.setAlignment(Qt.AlignCenter)

            QtWidgets.QApplication.processEvents()

    def open_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.video_source = file_name
            self.label_2.setText(f"选中视频: {os.path.basename(file_name)}")
        else:
            self.video_source = None
            self.label_2.setText("未选择视频")

    def toggle_camera(self):
        if self.video_source == 0:
            self.video_source = None
            self.label_2.setText("未选择视频")
            self.camera_2.setChecked(False)
            self.camera_2.setStyleSheet('')
        else:
            self.video_source = 0
            self.label_2.setText("使用摄像头")
            self.camera_2.setChecked(True)
            self.camera_2.setStyleSheet('background-color: lightgreen;')

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()
