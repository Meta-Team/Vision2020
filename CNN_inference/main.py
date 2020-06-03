
import random

import post_process 
import serial
import serial.tools.list_ports
import math

import cv2
import time
import numpy as np
import gluoncv
import gluoncv.data.transforms.image as timage
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import ndarray as nd
import os

# os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = '0'


############################# UART ####################################
test_uart = 1
uart_select = 1  # default 0, usb 1
have_uart = 0

port_list = list(serial.tools.list_ports.comports())
a = post_process.PostProcess()

dev = "/dev/ttyTHS1"
for i in port_list:
    print(i)
if len(port_list) > 0 and uart_select:
    dev = list(port_list[0])[0]
print('using:',dev)
    
# try to open UART
try:
    ser = serial.Serial(dev, 19200, timeout=10)
    have_uart = 1
except:
    print("open UART error!")

# send uart
def uart_send_post(bboxes,t):
    ledx,ledy,yaw_int,pitch_int = a.post_process(bboxes.reshape(-1,6),t)
    print('\033[0;31;47m', 'UART:',ledx,ledy,yaw_int,pitch_int, '\033[0m')
    # print('===== uart_send_data ==== ',arr)
    ser_data = bytes([ledx,32-ledy,0xFF,yaw_int // 100,yaw_int % 100,pitch_int // 100 ,pitch_int % 100, 0xFE ])
    if have_uart:
        try:
            ser.write(ser_data)
        except:
            print('send UART error!')
        
############################# TEST UART & POST ####################################
def testUART():
  while test_uart:
      for i in range(5):
          test_bboxes = np.array([[0.5,0.5,0.6,0.6,0,0.5],[0.55,0.55,0.6,0.6,0,0.1]])
          uart_send_post(test_bboxes,time.time())
          time.sleep(0.2)
      for i in range(5):
          test_bboxes = np.array([[0.4,0.4,0.5,0.5,0,0.5],[0.4,0.4,0.45,0.45,0,0.5]])
          uart_send_post(test_bboxes,time.time())
          time.sleep(0.2)

# testUART()

############################# CNN ####################################

# set the camera, on surface book we set camera_idx = 2
camera_idx = 0
cap = cv2.VideoCapture(camera_idx)
cap.open(camera_idx)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 120)

# batch_size is always 1 in real-time model inference
ctx = mx.gpu(0)
model_input_width, model_input_height = 416, 416
model_input_mean, model_input_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
model_filename = "YOLOv3_centernet_0_25_05_23_15_15/YOLOv3centernet0_25_30_128_5e-3.model_params"


# load CNN model
net = gluoncv.model_zoo.yolo3_mobilenet0_25_custom(classes=['red armor','blue armor'], transfer=None, pretrained_base=False, pretrained=False)


net.load_parameters(model_filename, ctx=ctx)


net.set_nms(post_nms=4)
net.hybridize()

# testUART()

print("Camera resolution is:", (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
print("Camera FPS is configured at:", cap.get(cv2.CAP_PROP_FPS))


for i in range(100000):
    t = time.time()
    
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = timage.imresize(nd.array(frame), model_input_width, model_input_height)
    img = mx.nd.image.to_tensor(img)
    img = mx.nd.image.normalize(img, mean=model_input_mean, std=model_input_std)
    #img = img[np.newaxis, :, :, :]
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    pred_cls, pred_score, pred_bbox = net(img.as_in_context(ctx))
    # temp settings for this model only, this strange number below is because a calibration mistake in YOLOv3 training
    pred_bbox =  pred_bbox / 416
    pred_cls, pred_score, pred_bbox = pred_cls.asnumpy(), pred_score.asnumpy(), pred_bbox.asnumpy()

    ######### send UART ##########
    # os.system('cls')
    # print('================= CNN out ===================')
    bboxes = np.hstack((pred_bbox[0],pred_cls[0], pred_score[0] ))
    # bboxes = (bboxes[np.lexsort(bboxes.T)])[-4:,:]   # sort by score
    # print('post process input\n', bboxes)
    uart_send_post(bboxes,t)
 

'''
# optional, print out the last inferenced image
_, ax = plt.subplots(figsize=(9,9), dpi=100)
img = img.as_in_context(mx.cpu(0))
gluoncv.utils.viz.plot_bbox(frame, pred_bbox[0], scores=pred_score[0], thresh=0.1,
                        labels=pred_cls[0], ax=ax, class_names=['red','blue'], colors={0:(1,0,0),1:(0,0,1)}, absolute_coordinates=False)
'''

