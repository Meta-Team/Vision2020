import cv2
import time
import numpy as np
import gluoncv
import gluoncv.data.transforms.image as timage
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import ndarray as nd

# batch_size is always 1 in real-time model inference
ctx = mx.gpu(0)
model_input_width, model_input_height = 416, 416
model_input_mean, model_input_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
model_filename = "YOLOv3_centernet_0_25_05_23_15_15/YOLOv3centernet0_25_30_128_5e-3.model_params"

# load CNN model
net = gluoncv.model_zoo.yolo3_mobilenet0_25_custom(classes=['red armor','blue armor'], transfer=None, pretrained_base=False, pretrained=False)
net.load_parameters(model_filename, ctx=ctx)
net.hybridize()

# set the camera
camera_idx = 0
cap = cv2.VideoCapture(camera_idx)
cap.open(camera_idx)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 120)
print("Camera resolution is:", (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
print("Camera FPS is configured at:", cap.get(cv2.CAP_PROP_FPS))

# print 10 sample images 
# for i in range(10):
#     ret, frame = cap.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     plt.figure()
#     plt.imshow(frame)
#     plt.show()
    

# do model inference in real-time for 120 frames, and test real inference FPS
# change loop from for loop to while loop after you have done post processing and is ready for consistent model inference
frames_total = 120
start_time = time.time()
num_frames = 0
for i in range(frames_total):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = timage.imresize(nd.array(frame), model_input_width, model_input_height)
    img = mx.nd.image.to_tensor(img)
    img = mx.nd.image.normalize(img, mean=model_input_mean, std=model_input_std)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    pred_cls, pred_score, pred_bbox = net(img.as_in_context(ctx))
    # convert pred_bbox from absolute coordinate to relative coordinate 
    pred_bbox =  pred_bbox / 416
    # convert pred_cls, pred_score, pred_bbox to numpy.ndarray
    pred_cls, pred_score, pred_bbox = pred_cls.asnumpy(), pred_score.asnumpy(), pred_bbox.asnumpy()
    # pred_cls is class of current bbox, -1 for background, 0 for red armor, and 1 for blue armor
    # pred_score is confidence score of current bbox, from 0 to 1
    # pred_bbox is position of bboxes, each row in the format of (xmin, ymin, xmax, ymax), in relative coordinate valued from 0 to 1
    # do post processing for net
    #TODO
    
    num_frames += 1

# After model inference for frames_total frames are done, calcuate total FPS and release camera device
time_total = time.time() - start_time
print("Camera input FPS is:", num_frames/time_total)
cap.release()

# optional, print out the last inferenced image
_, ax = plt.subplots(figsize=(9,9), dpi=100)
img = img.as_in_context(mx.cpu(0))
gluoncv.utils.viz.plot_bbox(frame, pred_bbox[0], scores=pred_score[0], thresh=0.1,
                           labels=pred_cls[0], ax=ax, class_names=['red','blue'], colors={0:(1,0,0),1:(0,0,1)}, absolute_coordinates=False)