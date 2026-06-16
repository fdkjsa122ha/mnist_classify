import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import numpy as np
from model import SimpleCNN
from cnn_train import test_loader
import pandas as pd
from sklearn import metrics
import seaborn as sns
from cnn_train import test_loader
import matplotlib.pyplot as plt 

# 加载模型
model = SimpleCNN()
model.load_state_dict(torch.load(r'CNN\mnist_cnn.pth'))  # 加载.pth文件
model.eval()  # 设置为评估模式

# 用来存储真实标签和预测标签
true_labels = []
predictions = []
probabilities = []

# 获取模型预测
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        
        # 获取预测的标签
        _, predicted = torch.max(outputs, 1)  # 获取预测的类别标签
        
        # 获取每个类别的概率
        probs = torch.softmax(outputs, dim=1)  # 获取预测的概率分布

        true_labels.extend(labels.numpy())  # 添加真实标签
        predictions.extend(predicted.numpy())  # 添加预测标签
        probabilities.extend(probs.numpy())  # 保存每个样本的所有类别的概率

# 计算精确率、召回率和AUC
precision = precision_score(true_labels, predictions, average='macro')
recall = recall_score(true_labels, predictions, average='macro')

# 对于多分类问题，AUC 需要传入每个类别的概率分布
auc = roc_auc_score(true_labels, np.array(probabilities), multi_class='ovr', average='macro')

# 输出结果
print(f'精确率 (Precision): {precision}')
print(f'召回率 (Recall): {recall}')
print(f'AUC (Area Under Curve): {auc}')

datafr = pd.DataFrame(metrics.confusion_matrix(true_labels,predictions,labels=[0,1,2,3,4,5,6,7,8,9]),columns=np.unique(true_labels),index=np.unique(predictions))
datafr.index.name = 'True Label'
datafr.columns.name = 'Predicted Label'
sns.heatmap(datafr, annot=True, fmt='d', cmap='Blues')
plt.show()