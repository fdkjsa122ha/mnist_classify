import joblib
from knn_train import test_x, test_y
import numpy as np
from sklearn import metrics
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 加载模型
loaded_model = joblib.load(r'KNN\knn_model.pkl')

# 使用加载的模型进行预测
y_pred = loaded_model.predict(test_x)
y_pred_proba = loaded_model.predict_proba(test_x)  
# cv2.namedWindow('test', cv2.WINDOW_NORMAL)
# cv2.imshow('test',test_x[3].reshape(28,28))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print('精确率是{}'.format(metrics.precision_score(test_y,y_pred,average='macro')))
print('召回率是{}'.format(metrics.recall_score(test_y,y_pred,average='macro')))
print('AUC是{}'.format(metrics.roc_auc_score(test_y,y_pred_proba, multi_class='ovr', average='macro')))

datafr = pd.DataFrame(metrics.confusion_matrix(test_y,y_pred,labels=[0,1,2,3,4,5,6,7,8,9]),columns=np.unique(test_y),index=np.unique(test_y))
datafr.index.name = 'True Label'
datafr.columns.name = 'Predicted Label'
sns.heatmap(datafr, annot=True, fmt='d', cmap='Blues')
plt.show()