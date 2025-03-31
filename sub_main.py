import cv2
import os
import sys
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from Vedio import Ui_Form  # 导入子 UI 界面
from ultralytics import YOLO

# YOLO 模型路径
MODEL_PATH = r"D:\python_projects\fire-recognition\weights\lz-train-base-yolo11n.pt"

class VideoThread(QThread):
    """ 视频检测线程 """
    update_frame = pyqtSignal(QtGui.QImage)  # 信号：用于更新 UI 界面的 label
    finished = pyqtSignal()  # 信号：检测完成
    error = pyqtSignal(str)  # 信号：错误提示

    def __init__(self, source=0):
        super().__init__()
        self.source = source  # 视频文件路径 或 摄像头ID（0 代表默认摄像头）
        self.running = True   # 控制线程运行状态
        self.model = YOLO(MODEL_PATH)  # 预加载 YOLO 模型

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error.emit("无法打开视频源！")
            return

        while self.running:
            ret, frame = cap.read()
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

            # 发送信号更新 UI
            self.update_frame.emit(qimg)
            time.sleep(0.03)  # 控制帧率，避免 CPU 过载

        cap.release()
        self.finished.emit()

    def stop(self):
        """ 停止线程 """
        self.running = False
        self.quit()
        self.wait()


class secondwindow(QtWidgets.QWidget, Ui_Form):
    """ 视频检测 UI 界面 """

    def __init__(self):
        super(secondwindow, self).__init__()
        self.setupUi(self)
        self.setWindowFlags(Qt.Window)  # 允许独立窗口显示
        self.thread = None  # 线程对象
        self.video_source = None  # 存储用户选择的视频源（文件路径 或 摄像头）

        # 绑定按钮事件
        self.video_input_2.clicked.connect(self.open_video)
        self.camera_2.clicked.connect(self.toggle_camera)
        self.det.clicked.connect(self.start_detection_button)  # 绑定开始检测按钮
        self.det_2.clicked.connect(self.stop_detection_button)  # 绑定停止检测按钮

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
        else:
            self.video_source = 0  # 选择摄像头
            self.label_2.setText("使用摄像头")

    def start_detection_button(self):
        """ 点击‘开始检测’按钮后，才真正启动检测 """
        if self.video_source is None:
            QMessageBox.warning(self, "警告", "请先选择视频文件或摄像头！")
            return

        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "警告", "检测已经在进行中，请先停止当前检测")
            return

        # 创建检测线程
        self.thread = VideoThread(self.video_source)
        self.thread.update_frame.connect(self.update_frame_label)
        self.thread.finished.connect(self.on_detection_finished)
        self.thread.error.connect(self.show_error)

        # 启动检测线程
        self.thread.start()

    def stop_detection_button(self):
        """ 停止视频检测 """
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread = None  # 释放线程对象
            QMessageBox.information(self, "提示", "检测已停止")
        else:
            QMessageBox.warning(self, "警告", "当前没有正在进行的检测")

    def update_frame_label(self, qimg):
        """ 更新 UI 界面的 `frame_label_2` """
        self.frame_label_2.setPixmap(QPixmap.fromImage(qimg))

    def on_detection_finished(self):
        """ 处理检测完成逻辑 """
        QMessageBox.information(self, "提示", "视频检测完成")

    def show_error(self, message):
        """ 处理错误信息 """
        QMessageBox.critical(self, "错误", message)

    def closeEvent(self, event):
        """ 关闭窗口时，确保线程退出 """
        if self.thread and self.thread.isRunning():
            self.thread.stop()
        event.accept()
