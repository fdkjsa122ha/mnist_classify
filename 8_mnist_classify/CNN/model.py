import torch.nn as nn
import torch.nn.functional as F

# 定义我们自己的神经网络类
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一层卷积：输入1个通道（灰度图），输出10个特征图，卷积核大小5x5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 第二层卷积：输入10个特征图，输出20个特征图，卷积核大小5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 全连接层：把提取到的特征展开，变成一维数据，然后进行分类
        self.fc1 = nn.Linear(320, 50)
        # 输出层：输出10个数字（0-9的概率分布）
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # 数据正向传播的过程
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # 卷积 -> 激活 -> 池化
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # 卷积 -> 激活 -> 池化
        x = x.view(-1, 320)                        # 展平图像数据
        x = F.relu(self.fc1(x))                    # 全连接 -> 激活
        x = self.fc2(x)                            # 输出结果
        return F.log_softmax(x, dim=1)             # 转换为概率