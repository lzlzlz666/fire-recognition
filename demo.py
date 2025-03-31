import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from GUI import Ui_MainWindow
from ultralytics import YOLO

from sub_main import secondwindow    # 导入子UI类

# 火焰识别模型
MODEL_PATH_FIRE = r"D:\python_projects\fire-recognition\weights\lz-train-base-yolo11n.pt"
# 通用识别模型
MODEL_PATH_YOLO = r"D:\python_projects\fire-recognition\weights\yolo11n.pt"

class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)

        self.second_window = secondwindow()     # 实例化子界面
        self.actionvedio.triggered.connect(self.pushbutton_function)

        self.uploadPushButton.clicked.connect(self.open_image)

        # 火焰识别
        self.pushButton_3.clicked.connect(lambda: self.detect_and_show(MODEL_PATH_FIRE))
        # 通用识别
        self.pushButton_2.clicked.connect(lambda: self.detect_and_show(MODEL_PATH_YOLO))

        self.current_pixmap = None  # 存储当前显示的图像
        self.file_name = None  # 存储上传的图像的路径

    def pushbutton_function(self):
        self.second_window.show()  # 通过点击按钮弹出子界面
    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.bmp *.jpeg)")
        if file_name:
            pixmap = QPixmap(file_name)
            self.file_name = file_name

            # 按比例缩放图片
            pixmap = pixmap.scaled(self.label_1.width(), self.label_1.height(), Qt.KeepAspectRatio)

            # 保存并显示图片
            self.current_pixmap = pixmap
            self.label_1.setPixmap(pixmap)
            self.label_1.setAlignment(Qt.AlignCenter)

    def detect_and_show(self, model_path):
        if self.file_name is None:
            QMessageBox.warning(self, "警告", "没有选择图片")
            return

        # 加载 YOLO 模型
        model = YOLO(model_path)

        # 进行预测
        results = model.predict(self.file_name, save=True)

        # 获取 YOLO 生成的最新检测结果目录
        result_dir = self.get_latest_predict_dir("runs/detect")
        if result_dir is None:
            QMessageBox.warning(self, "警告", "未找到检测结果文件夹")
            return

        # 构造检测结果图片的路径
        result_img_name = os.path.splitext(os.path.basename(self.file_name))[0] + '.jpg'  # 确保后缀为 .jpg
        result_img_path = os.path.join(result_dir, result_img_name)

        # 读取并显示结果图片
        if os.path.exists(result_img_path):
            pixmap = QPixmap(result_img_path)
            pixmap = pixmap.scaled(self.label_2.width(), self.label_2.height(), Qt.KeepAspectRatio)
            self.label_2.setPixmap(pixmap)
            self.label_2.setAlignment(Qt.AlignCenter)
        else:
            QMessageBox.warning(self, "警告", f"未找到结果图像: {result_img_path}")

    def get_latest_predict_dir(self, base_dir):
        """获取最新的 YOLO 预测结果文件夹"""
        if not os.path.exists(base_dir):
            return None
        subdirs = [d for d in os.listdir(base_dir) if d.startswith("predict")]
        if not subdirs:
            return None
        # 按照 predict、predict1、predict2 这样的顺序排列，选最新的
        subdirs.sort(key=lambda x: int(x.replace("predict", "")) if x.replace("predict", "").isdigit() else 0, reverse=True)
        return os.path.join(base_dir, subdirs[0])


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())