import os
import json
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
%matplotlib inline

#%%
def is_light_bar(image, avg_threshold=0.5*255):
    '''判断给的图片是不是纯灯条，判断方法：1.长宽比，2.图像红蓝通道平均值'''
    if (image[:, :, 0].mean() > avg_threshold or image[:, :, 2].mean() > avg_threshold):
        return True
    return False


def move_light_bar(image_file, region_folder):
    '''去除图像文件夹中的图片，同时去除annotation文件中的记录，把这些挑选出来的放在其他的文件中，待审核'''
    pass


def show_light_bar():
    '''用于测试，显示不同通道平均值选出的图片样本'''
    root_path = ''
    num_img = 0
    for image_file in os.listdir(root_path):
        image = cv2.imread(image_file)
        if is_light_bar(image):
            plt.subplot(5,5,num_img)
            plt.imshow(image)
            num_img += num_img
        if num_img > 25:
            break
    plt.show()

show_light_bar()
#%%
if __name__ == "__main__":
    root_path = ''  # TODO change to your own path
    regions = [x for x in os.listdir(root_path)
               if os.path.isdir(os.path.join(root_path, x))]
    for region in regions:
        image_files = os.listdir(os.path.join(root_path, region, 'image'))
        for image_file in image_files:
            image = cv2.imread(os.path.join(
                root_path, region, 'image', image_file))
            if is_light_bar(image):
                move_light_bar(image_file, os.path.join(root_path, region))
