
"""
    使用python实现：读取USB摄像头的画面
"""
# 导入CV2模块
import cv2
import os
import datetime

camera_idx = 0
cap = cv2.VideoCapture(camera_idx)
cap.open(camera_idx)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

def read_usb_capture():
    # 添加这句是可以用鼠标拖动弹出的窗体
    cv2.namedWindow('real_img', cv2.WINDOW_NORMAL)

    # .mp4格式 , 25为 FPS 帧率， （640,480）为大小
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('temp.mp4', fourcc, 25, (640, 480))

    while(cap.isOpened()):
        # 读取摄像头的画面
        ret, frame = cap.read()

        # 进行写操作
        out.write(frame)
        # 真实图
        cv2.imshow('real_img', frame)
        # 按下'esc'就退出
        if cv2.waitKey(1) & 0xFF == 27:
            break
    # 释放画面
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    read_usb_capture() # 启动摄像
    name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 当前的时间
    os.remove('temp.mp4') # 删除中间视频文件
