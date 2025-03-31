import cv2
import os
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from Vedio import Ui_Form  # 导入子 UI 界面
from ultralytics import YOLO

# 默认 YOLO 模型路径
MODEL_PATH = r"D:\python_projects\fire-recognition\weights\lz-train-base-yolo11n.pt"
# 通用 YOLO 模型路径
MODEL_PATH_YOLO = r"D:\python_projects\fire-recognition\weights\yolo11n.pt"


class secondwindow(QtWidgets.QWidget, Ui_Form):
    """ 视频检测 UI 界面 """

    def __init__(self):
        super(secondwindow, self).__init__()
        self.setupUi(self)
        self.setWindowFlags(Qt.Window)  # 允许独立窗口显示
        self.model = YOLO(MODEL_PATH)  # 默认 YOLO 模型
        self.video_source = None  # 存储用户选择的视频源（文件路径 或 摄像头）

        # 绑定按钮事件
        self.video_input_2.clicked.connect(self.open_video)
        self.camera_2.clicked.connect(self.toggle_camera)
        self.det.clicked.connect(self.start_detection_button)  # 绑定开始检测按钮
        self.det_2.clicked.connect(self.stop_detection_button)  # 绑定停止检测按钮
        self.det_3.clicked.connect(self.start_detection_button_with_yolo)  # 绑定新的 YOLO 模型按钮

        self.cap = None  # 视频捕获对象

    def open_video(self):
        """ 选择视频文件（但不立即检测）"""
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.video_source = file_name  # 存储选择的视频文件
            self.label_2.setText(f"选中视频: {os.path.basename(file_name)}")  # 显示选中的文件名
        else:
            self.video_source = None  # 清除选择
            self.label_2.setText("未选择视频")

    def toggle_camera(self):
        """ 选择/取消摄像头 """
        if self.video_source == 0:
            self.video_source = None  # 取消摄像头选择
            self.label_2.setText("未选择视频")
            self.camera_2.setChecked(False)
            self.camera_2.setStyleSheet('')  # 重置按钮样式，取消高亮
        else:
            self.video_source = 0  # 选择摄像头
            self.label_2.setText("使用摄像头")
            self.camera_2.setChecked(True)
            self.camera_2.setStyleSheet('background-color: lightgreen;')  # 设置按钮高亮（例如绿色背景）

    def start_detection_button(self):
        """ 点击‘开始检测’按钮后，才真正启动检测 """
        if self.video_source is None:
            QMessageBox.warning(self, "警告", "请先选择视频文件或摄像头！")
            return

        # 如果已经在检测中，提醒用户
        if self.cap and self.cap.isOpened():
            QMessageBox.warning(self, "警告", "检测已经在进行中，请先停止当前检测")
            return

        # 打开视频源
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开视频源！")
            return

        # 开始处理视频帧
        self.process_video()

    def start_detection_button_with_yolo(self):
        """ 点击‘开始检测’按钮后，使用通用YOLO模型进行检测 """
        self.model = YOLO(MODEL_PATH_YOLO)  # 切换到通用YOLO模型
        self.start_detection_button()  # 调用原始的检测方法

    def stop_detection_button(self):
        """ 停止视频检测 """
        if self.cap and self.cap.isOpened():
            self.cap.release()  # 释放视频资源
            self.cap = None  # 清除视频捕获对象
            self.frame_label_2.clear()  # 清除显示的图像
            QMessageBox.information(self, "提示", "检测已停止")

            # 停止后重置模型为火焰检测模型
            self.model = YOLO(MODEL_PATH)  # 重置回默认火焰检测模型
        else:
            QMessageBox.warning(self, "警告", "当前没有正在进行的检测")

    def process_video(self):
        """ 处理视频帧并更新 UI """
        while self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # 进行目标检测
            results = self.model(frame)
            for result in results:
                frame = result.plot()  # 在帧上绘制检测框

            # 将 OpenCV 图像转换为 PyQt 格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 更新 UI
            self.frame_label_2.setPixmap(QPixmap.fromImage(qimg))

            QtWidgets.QApplication.processEvents()  # 处理 UI 事件，避免界面冻结

    def closeEvent(self, event):
        """ 关闭窗口时，确保视频源被释放 """
        if self.cap and self.cap.isOpened():
            self.cap.release()  # 释放视频捕获对象
        event.accept()
