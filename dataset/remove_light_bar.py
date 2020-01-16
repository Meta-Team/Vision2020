# -*- coding: utf-8 -*-
# %%
import os
import json
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import shutil

main_root_path = '/mnt/e/robomaster/mydump/'
# %%
def is_light_bar(image,ratio=1.3,red_avg_threshold=0.2*255,blue_avg_threshold=0.2*255):
    '''判断给的图片是不是纯灯条，判断方法：
        1.长宽比，2.图像红蓝通道平均值, 3.图像中最长直线长度'''
    try:
        # 判断长宽比和平均值
        if (image.shape[0] / image.shape[1] > 2*ratio):
            return True
        if (image.shape[0] / image.shape[1] > ratio and
            (image[:, :, 0].mean() > red_avg_threshold or
             image[:, :, 2].mean() > blue_avg_threshold)):

            '''方法一：Hough直线检测（准确率一般）'''
            # min_length = 20  # 直线长度阈值
            # imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度化
            # edge = cv2.Canny(imgray, 50, 180, apertureSize=3)  # 边缘检测
            # lines = cv2.HoughLinesP(edge, 1, np.pi/180, min_length)  # 直线检测

            '''方法二：除去白色的Hough直线检测（感觉比方法一更准）'''
            lines = []
            min_length = int(image.shape[0]*0.8)
            imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 灰度化
            _, mask = cv2.threshold(imgray, 175, 255, cv2.THRESH_BINARY_INV) # 二值化，白色高光变0

            if(image[:, :, 0].mean() > red_avg_threshold):
                _, imred = cv2.threshold(
                    image[:, :, 0], 125, 255, cv2.THRESH_BINARY) # 红色通道阈值化处理
                imred = cv2.bitwise_and(imred, mask)    # 蒙版，删除高光区域
                lines.extend(cv2.HoughLinesP(imred, 1, np.pi/180, min_length)) #检测纯红色区域有无较长直线

            if(image[:, :, 2].mean() > blue_avg_threshold):
                _, imblue = cv2.threshold(
                    image[:, :, 2], 125, 255, cv2.THRESH_BINARY)
                imblue = cv2.bitwise_and(imblue, mask)
                lines.extend(cv2.HoughLinesP(imblue, 1, np.pi/180, min_length))
            
            if lines is not None: # 存在较长直线，说明有灯条（或者图片不清晰，也删了）
                return True
        return False
    except:
        # TODO:把不能读取的图片删除
        return False


def remove_light_bar(image_file, root_path, removed_anno):
    '''去除图像文件夹中的图片，同时去除annotation文件中的记录，把这些挑选出来的放在其他的文件中，待审核'''
    # print('remove_light_bar : %s' % (image_file))
    anno_name = '_'.join(image_file.split('_')[:-1])+'.json'
    try:
        with open(os.path.join(root_path, 'image_annotation', anno_name),
                  'r') as anno_file:
            json_dict = json.load(anno_file)
        shutil.move(os.path.join(root_path, 'image', image_file),
                    os.path.join(root_path, 'lightbar', image_file))
        ret = json_dict[image_file]
        del json_dict[image_file]
        with open(os.path.join(root_path, 'image_annotation', anno_name), 'w') as anno_file:
            json.dump(json_dict, anno_file, ensure_ascii=False)
        return ret
    except Exception as e:
        print(e)


def show_light_bar(root_path, row=5, col=5, batch=2):
    '''用于测试，显示灯条的图片样本'''
    num_img = 1
    num_batch = 1
    image_folder = os.path.join(root_path, 'image')
    for image_file in os.listdir(image_folder):
        image = cv2.imread(os.path.join(image_folder, image_file))
        if is_light_bar(image):
            plt.subplot(row, col, num_img)
            plt.axis('off')
            plt.imshow(image)
            num_img += 1
        if num_img > row*col:
            plt.show()
            num_img = 1
            num_batch += 1
        if num_batch > batch:
            break
    plt.show()
    return

# gui可能会用到的tkinter接口
    # img_open = Image.open('img_png.png')
    # img_png = ImageTk.PhotoImage(img_open)
    # label_img = Tkinter.Label(root, image = img_png)
    # label_img.pack()


def main_computer(root_path):
    # 打开anno文件，接着上一次的做
    try:
        with open(os.path.join(root_path, 'lightbar', 'removed_anno.json'), 'r') as anno_file:
            removed_anno = json.load(anno_file)
            anno_file.close()
    except:
        # 不存在的话说明是新的文件夹，新建一个
        print('start new: %s' % root_path)
        removed_anno = {}
    
    image_folder = os.path.join(root_path, 'image')
    lightbar_folder = os.path.join(root_path, 'lightbar')
    # 建立灯条文件夹，放灯条图片，备用
    if not os.path.exists(lightbar_folder):
        os.makedirs(lightbar_folder)
    # 遍历图片，检测灯条，是灯条就移走
    for image_file in os.listdir(image_folder):
        image = cv2.imread(os.path.join(image_folder, image_file))
        if is_light_bar(image):
            removed_anno[image_file] = remove_light_bar(
                image_file, root_path, removed_anno)
    # 把移走的灯条记录到json文件中
    with open(os.path.join(root_path, 'lightbar', 'removed_anno.json'), 'w') as anno_file:
        json.dump(removed_anno, anno_file, ensure_ascii=False)
        anno_file.close()

# %%


if __name__ == "__main__":
    '''筛选出是灯条的照片'''
    regions = [x for x in os.listdir(main_root_path) if (
        os.path.isdir(os.path.join(main_root_path, x))and x != 'lightbar')]
    for region in regions:
        root_path = os.path.join(main_root_path, region)
        print('working on %s' % region)
        '''二选一，测试还是运行'''
        # show_light_bar(root_path, 1, 5, 30) # test
        
        main_computer(root_path) #run

# %%
