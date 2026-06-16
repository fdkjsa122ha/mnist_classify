import cv2
import torch
from model import SimpleCNN
from cnn_train import transform
from pre_process import pre_process


if __name__ == '__main__':
    image_path = r"F:\1_project_store\2_dev_info\3_data_set\6_machine_learning\8_mnist_classify\fig_1.jpg"
    digit=pre_process(cv2.imread(image_path,cv2.IMREAD_GRAYSCALE))
    img_tensor=transform(digit).unsqueeze(0)  # 添加批次维度
    model = SimpleCNN()
    # 读取我们刚才 train.py 保存下来的权重
    model.load_state_dict(torch.load(r'8_mnist_classify\CNN\mnist_cnn.pth'))
    model.eval()  # 设置为测试模式（关闭Dropout等）
    with torch.no_grad():  # 测试时不需要计算梯度
        output = model(img_tensor)
        # 获取概率最大的那个数字的索引
        prediction = output.argmax(dim=1, keepdim=True).item()
    print("预测结果:", prediction)
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test',digit.reshape(28,28))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
