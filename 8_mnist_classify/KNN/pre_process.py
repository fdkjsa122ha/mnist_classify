import cv2
import numpy as np 
import matplotlib.pyplot as plt
from icecream import ic

def pre_process(img):
    # if img is None:
    #     print("图像加载失败，请检查文件路径")
    # else:
    #     print("图像加载成功")
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    img_canny=cv2.Canny(img_blur, 100, 200)
    h=img.shape[0]
    w=img.shape[1]
    val_h=np.zeros(h,dtype=int)
    val_w=np.zeros(w,dtype=int)
    for i in range(h):
        for j in range(w):
            val_h[i]+=img_canny[i][j]
    for i in range(w):
        for j in range(h):
            val_w[i]+=img_canny[j][i]
    h_upper=np.argmax(val_h)
    h_lower=np.argmax(val_h)
    w_upper=np.argmax(val_w)
    w_lower=np.argmax(val_w)
    # 保证边界不会越界，避免h_lower、h_upper、w_lower、w_upper越界。
    while h_lower > 1 and val_h[h_lower] > 0:
        h_lower -= 1

    while h_upper < len(val_h) - 2 and val_h[h_upper] > 0:
        h_upper += 1

    while w_lower > 1 and val_w[w_lower] > 0:
        w_lower -= 1

    while w_upper < len(val_w) - 2 and val_w[w_upper] > 0:
        w_upper += 1

    delta=w_upper-w_lower-(h_upper-h_lower)# 裁剪出正方形

    # 计算裁剪区域的上下边界
    if delta > 0:
        # 对于delta大于0的情况
        start_h = max(0, h_lower - delta // 2 - 7)  # 防止越界，确保start_h >= 0
        end_h = min(h, h_upper + delta - delta // 2 + 7)  # 防止越界，确保end_h <= img.height
        start_w = max(0, w_lower - 7)  # 防止越界，确保start_w >= 0
        end_w = min(w, w_upper + 7)  # 防止越界，确保end_w <= img.width
    else:
        # 对于delta小于等于0的情况
        start_h = max(0, h_lower - 7)  # 防止越界，确保start_h >= 0
        end_h = min(h, h_upper + 7)  # 防止越界，确保end_h <= img.height
        start_w = max(0, w_lower + delta // 2 - 7)  # 防止越界，确保start_w >= 0
        end_w = min(w, w_upper - delta + delta // 2 + 7)  # 防止越界，确保end_w <= img.width

    # 使用计算后的边界进行图像裁剪
    tailor_img = img[start_h:end_h, start_w:end_w]
    if tailor_img is None or tailor_img.size == 0:
        print("Error: Image is empty after processing.")
    else:
        tailor_img=cv2.resize(tailor_img,(28,28))
    # print("裁剪后图像的形状:", tailor_img.shape)
    for i in range(28):
        for j in range(28):
            if tailor_img[i][j]>100:
                tailor_img[i][j]=0
            else:
                tailor_img[i][j]=255-tailor_img[i][j]
    return tailor_img
# plt.bar([i for i in range(h)],val_h)
# plt.show()


if __name__ == '__main__':
    path=r"F:\1_project_store\2_dev_info\3_data_set\6_machine_learning\8_mnist_classify\fig_2.png"
    tailor_img=pre_process(cv2.imread(path,cv2.IMREAD_GRAYSCALE))
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test',tailor_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()