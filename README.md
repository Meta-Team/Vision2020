## Introduction

YOLOv3 application in Robomaster, based on https://github.com/ultralytics/yolov3.

YOLOv3 is an one-step real-time object detector. YOLOv3-tiny can run as fast as 20 fps on jetson-nano, but will lose some accuracy.  

## Environment

- Windows10/Linux
- cuda, cudnn, python, pytorch

## Training 

*see original tutorial for more information*

1. put images in data/images
2. put .xml files in data/Annotations
3. convert .xml files into required format
   `cd data`
   `python convert_labels.py`
   
4. generate *train.txt* and *valid.txt*
   `python generate_sets.py`
5. modify *data/robot.data*, *robot.names*, *yolov3-tiny.cfg* if necessary
5. train
   `cd ..`
   `python train.py --data data/robot.data --cfg yolov3-tiny.cfg --weights yolov3-tiny.conv.15 --epochs 30`
6. test
   `python detect.py --cfg yolov3-tiny.cfg --weights weights/best.pt --names data/robot.names --source test_set/input --output test_set/output`

**TODO**

- [ ] convert trained model to onnx. https://github.com/marvis/pytorch-caffe-darknet-convert

There are many implementations of YOLOv3 on github.

- official: [darknet](https://github.com/pjreddie/darknet), a neural network framework in C, more than YOLO, easy to train and use.
- pytorch: https://github.com/ultralytics/yolov3. I use this. 
- tensorflow-keras: https://github.com/qqwweee/keras-yolo3
- Caffe: https://github.com/lewes6369/TensorRT-Yolov3

## Detection

**TODO**

- [ ] Tensor RT is twice as fast as darknet, it support onnx and tensorflow model. There are official demos from NVIDIA. Use Tensor RT to do the detection on Jetson-Nano. It has both python/C++ API.
\
   some resources: 
   https://github.com/Cw-zero/TensorRT_yolo3
   https://jkjung-avt.github.io/tensorrt-yolov3
   https://github.com/enazoe/yolo-tensorrt
   

## Others

- [ ] amors are too small in the original pictures. It needs rescaling.

- [ ] pruning. https://github.com/coldlarry/YOLOv3-complete-pruning
