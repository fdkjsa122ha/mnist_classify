import cv2
import torch
from torch import nn
import sys
sys.path.append(r'F:\1_project_store\1_代码_study\6_机器学习\8_mnist_classify')# 会在该路径下寻找KNN模块
from KNN.pre_process import pre_process
from model import SimpleCNN

def open_camera():
    # 加载 .pth 模型
    model = SimpleCNN()
    model.load_state_dict(torch.load(r'CNN\mnist_cnn.pth'))  # 加载模型参数
    model.eval()  # 设置为评估模式（非常重要！）

    cap = cv2.VideoCapture(0)
    # 设置摄像头的分辨率为 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
    
    temp = -1
    cv2.namedWindow('my image', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法打开摄像头")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将图像转为灰度图
        
        # 预处理输入图像
        digit = pre_process(frame)
        
        # 将图像转换为 PyTorch 张量
        digit_tensor = torch.tensor(digit, dtype=torch.float32)
        digit_tensor = digit_tensor.unsqueeze(0).unsqueeze(0)  # 扩展维度，以适应模型输入 (1, 1, 28, 28)

        # 进行预测
        with torch.no_grad():  # 关闭梯度计算，减少内存使用
            output = model(digit_tensor)  # 模型的前向传播
            _, predicted = torch.max(output, 1)  # 获取预测结果（最大概率的类）
        
        # 输出预测结果
        if temp != predicted.item():
            print("预测结果:", predicted.item())  # 显示预测的类别
            temp = predicted.item()

        # 显示处理后的图像
        cv2.imshow('my image', digit)
        cv2.imshow('Camera', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    open_camera()
    cv2.destroyAllWindows()