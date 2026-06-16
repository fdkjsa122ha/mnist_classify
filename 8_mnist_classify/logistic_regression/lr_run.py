import cv2
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack
import joblib
from pre_process import pre_process

if __name__ == '__main__':
    # 加载模型
    loaded_model = joblib.load(r'logistic_regression\logistic_model.pkl')
    image_path = r"F:\1_project_store\2_dev_info\3_data_set\6_machine_learning\real_programming\fig_2.png"
    digit=pre_process(cv2.imread(image_path,cv2.IMREAD_GRAYSCALE))/255
    print("预测结果:", loaded_model.predict(digit.reshape(1, -1)))
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test',digit.reshape(28,28))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
