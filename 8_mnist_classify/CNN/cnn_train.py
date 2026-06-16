import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from model import SimpleCNN
from struct import unpack
from icecream import ic
import os

# 手动加载 MNIST 数据（保持不变）
def loadmnist(imagefile, labelfile):
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    images.read(4)  # 跳过 magic number
    number_of_images = unpack('>I', images.read(4))[0]
    rows = unpack('>I', images.read(4))[0]
    cols = unpack('>I', images.read(4))[0]

    labels.read(4)
    N = unpack('>I', labels.read(4))[0]

    x = np.zeros((N, rows * cols), dtype=np.uint8)
    y = np.zeros(N, dtype=np.uint8)
    for i in range(N):
        for j in range(rows * cols):
            tmp_pixel = unpack('>B', images.read(1))[0]
            x[i][j] = tmp_pixel
        tmp_label = unpack('>B', labels.read(1))[0]
        y[i] = tmp_label

    images.close()
    labels.close()
    return x, y

# 自定义 Dataset 类，包装 NumPy 数据并应用 transform
class MNISTDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 原始数据是扁平向量 (784,)，需要 reshape 为 (28, 28)
        img = self.data[idx].reshape(28, 28).astype(np.uint8)
        target = int(self.targets[idx])

        if self.transform:
            img = self.transform(img)  # transform 会将 PIL 或 ndarray 转为 Tensor 并标准化
        return img, target

# 设置数据集路径
data_set_path = r"F:\1_project_store\2_dev_info\3_data_set\6_machine_learning\1_logic_regression_MNIST\dataset"

# 图像预处理：转为 Tensor（自动添加通道维度并缩放到 [0,1]），然后标准化
transform = transforms.Compose([
    transforms.ToTensor(),                # 将 numpy.ndarray (H x W) 转为 Tensor (1 x H x W) 并归一化到 [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # 使用 MNIST 数据集的均值和标准差
])

# 加载原始数据
train_x, train_y = loadmnist(
    os.path.join(data_set_path, 'train-images-idx3-ubyte'),
    os.path.join(data_set_path, 'train-labels-idx1-ubyte')
)
test_x, test_y = loadmnist(
    os.path.join(data_set_path, 't10k-images-idx3-ubyte'),
    os.path.join(data_set_path, 't10k-labels-idx1-ubyte')
)

# 构建完整的训练 Dataset 和测试 Dataset
full_train_dataset = MNISTDataset(train_x, train_y, transform=transform)
test_dataset = MNISTDataset(test_x, test_y, transform=transform)

# 划分训练集和验证集
train_size = 50000
val_size = 10000
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 接下来可以使用 train_loader, val_loader, test_loader 进行模型训练和评估


def train(train_data,val_data,epochs=20):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_data):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_data:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += F.cross_entropy(output, target, reduction='sum').item()  # 累加损失
                pred = output.argmax(dim=1, keepdim=True)  # 获取预测结果
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_data.dataset)  # 平均损失
        accuracy = 100. * correct / len(val_data.dataset)
        print(f"第 {epoch + 1}/{epochs} 轮完成 | 验证集损失: {val_loss:.4f} | 验证集准确率: {accuracy:.2f}%")

    torch.save(model.state_dict(), r'CNN\mnist_cnn.pth')
    print("\n模型训练完成，已保存为 mnist_cnn.pth\n")



if __name__ == '__main__':
    train(train_loader,val_loader)