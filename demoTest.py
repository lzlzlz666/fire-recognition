import torch
from ultralytics import YOLO
import cv2
import os

# 固定图片和模型路径（如果不希望每次输入）
IMAGE_PATH = r"D:\python_projects\fire-recognition\test-imgs\fire.jpg"
MODEL_PATH = r"D:\python_projects\fire-recognition\weights\lz-train-base-yolo11n.pt"

def detect_and_show(image_path, model_path):
    # 加载 YOLO 模型
    model = YOLO(model_path)

    # 进行预测并保存结果
    results = model.predict(image_path, save=True)

    # 获取 YOLO 生成的检测结果路径
    result_dir = results[0].save_dir if hasattr(results[0], "save_dir") else "runs/detect/predict"
    result_img_path = os.path.join(result_dir, os.path.basename(image_path))

    # 读取并显示结果图片
    if os.path.exists(result_img_path):
        img = cv2.imread(result_img_path)
        cv2.imshow("Detection Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"未找到结果图像: {result_img_path}")

if __name__ == "__main__":
    detect_and_show(IMAGE_PATH, MODEL_PATH)
