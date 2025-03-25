# 🔥 Fire Recognition with YOLOv11n

基于 **YOLOv11n** 训练的火焰识别模型，可快速检测图像或视频中的火灾区域，适用于火灾预警等场景。  

## 📌 项目简介  

本项目使用 **YOLOv11n**（轻量级 YOLO 版本）进行 **火焰检测**，通过标注火焰数据集并进行训练，得到 `lz-train-base-yolo11n.pt` 作为最终检测模型。  

## 📂 目录结构  

```plaintext
fire-recognition/
├── weights/                # 训练好的模型权重
│   ├── lz-train-base-yolo11n.pt
├── test-imgs/              # 测试图片存放目录
│   ├── fire.jpg
├── demo.py                 # 运行 YOLO 检测的 Python 脚本
├── README.md               # 本说明文件
