import cv2
import joblib
from pre_process import pre_process


def open_camera():
    loaded_model = joblib.load(r'KNN\knn_model.pkl')
    cap = cv2.VideoCapture(0)
    # 设置摄像头的分辨率为 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 100)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)
    temp=-1
    cv2.namedWindow('my image', cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法打开摄像头")
            break
        # digit=frame
        frame1=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        digit = pre_process(frame1)
        v=loaded_model.predict(digit.reshape(1, -1))
        if temp!=v: 
            print("预测结果:", v)
            temp=v
        cv2.imshow('my image', digit)
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    open_camera()
    cv2.destroyAllWindows()