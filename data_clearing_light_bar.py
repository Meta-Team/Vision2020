import os
import json
import cv2
import numpy
import json


def is_light_bar(image):
    '''判断给的图片是不是纯灯条，判断方法：1.长宽比，2.图像红蓝通道平均值'''
    pass

def delete(image_file):
    '''删除图像文件夹中的图片，同时删除annotation文件中的记录'''
    pass

if __name__ == "__main__":
    root_path = ''  #TODO change to your own path
    regions = [x for x in os.listdir(root_path)
               if os.path.isdir(os.path.join(root_path, x))]
    for region in regions:
        image_files = os.listdir(os.path.join(root_path, region, 'image'))
        for image_file in image_files:
            image = cv2.imread(os.path.join(root_path, region,'image',image_file))
            if is_light_bar(image):
                delete(image_file)