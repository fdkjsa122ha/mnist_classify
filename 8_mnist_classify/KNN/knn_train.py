from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from struct import unpack
import os
import joblib
from matplotlib import pyplot as plt

def loadmnist(imagefile, labelfile):
    # Open the images with gzip in read binary mode
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)
    N = labels.read(4)
    N = unpack('>I', N)[0]

    # Get data
    x = np.zeros((N, rows*cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros(N, dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        for j in range(rows*cols):
            tmp_pixel = images.read(1)  # Just a single byte
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            x[i][j] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]

    images.close()
    labels.close()
    return (x, y)

# 6万条训练数据，1万条测试数据
data_set_path=r"F:\1_project_store\2_dev_info\3_data_set\6_machine_learning\1_logic_regression_MNIST\dataset"
train_x, train_y = loadmnist(os.path.join(data_set_path,'train-images-idx3-ubyte'),os.path.join(data_set_path,'train-labels-idx1-ubyte'))
test_x, test_y = loadmnist(os.path.join(data_set_path,'t10k-images-idx3-ubyte'),os.path.join(data_set_path,'t10k-labels-idx1-ubyte'))
train_x = train_x / 255.0
test_x = test_x / 255.0

model =KNeighborsClassifier(n_neighbors=10)
model.fit(train_x, train_y)

# model = LogisticRegression(C=50.0 / len(train_x), penalty="l1", solver="saga", tol=0.01, multi_class="ovr", max_iter=100)
# model.fit(train_x,train_y)
joblib.dump(model, r'KNN\knn_model.pkl')

