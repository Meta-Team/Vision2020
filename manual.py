
import os
import json
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import shutil


def is_light_bar(image,
                 ratio=1.5,
                 red_avg_threshold=0.3*255,
                 blue_avg_threshold=0.3*255):
    '''判断给的图片是不是纯灯条，判断方法：
        1.长宽比，2.图像红蓝通道平均值, 3.图像边缘直线长度'''
    try:
        # 判断长宽比和平均值
        if (image.shape[0] / image.shape[1] > ratio and
            (image[:, :, 0].mean() > red_avg_threshold or
             image[:, :, 2].mean() > blue_avg_threshold)):
            # Hough直线检测
            imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度化
            edge = cv2.Canny(imgray, 50, 180, apertureSize=3)  # 边缘检测
            min_length = 20  # 直线长度阈值
            lines = cv2.HoughLinesP(edge, 1, np.pi/180, min_length)  # 直线检测
            if lines is not None:
                return True
        return False
    except Exception as e:
        print(e)
        # TODO:把不能读取的图片删除
        return False


def remove_light_bar(image_files, root_path, num_img, removed_anno):
    '''去除图像文件夹中的图片，同时去除annotation文件中的记录，把这些挑选出来的放在其他的文件中，待审核'''
    print('remove_light_bar : %s' % (image_files[num_img]))
    image_name = image_files[num_img]
    anno_name = '_'.join(image_name.split('_')[:-1])+'.json'
    with open(os.path.join(root_path, 'image_annotation', anno_name),
              'r') as anno_file:
        json_dict = json.load(anno_file)
    try:
        shutil.move(os.path.join(root_path, 'image', image_name),
                    os.path.join(main_root_path, 'lightbar', image_name))
        removed_anno[image_name] = json_dict[image_name]
        del json_dict[image_name]
    except Exception as e:
        print(e)
    with open(os.path.join(root_path, 'image_annotation', anno_name), 'w') as anno_file:
        json.dump(json_dict, anno_file, ensure_ascii=False)


'''下面这几个函数都是用于gui人工选图'''


def change_image():
    global image_files, root_path, num_img
    while(1):
        image = cv2.imread(os.path.join(
            root_path, 'image', image_files[num_img]))
        if is_light_bar(image):
            break
        num_img += 1
    img.set_data(image)
    fig.canvas.draw_idle()


def onclick(event):
    '''回到上一张'''
    if event.inaxes != ax:
        return
    global image_files, root_path, num_img
    num_img -= 1
    print('at image %s' % (num_img))
    image = cv2.imread(os.path.join(root_path, 'image', image_files[num_img]))
    img.set_data(image)
    fig.canvas.draw_idle()
    return


def on_key_press_n(event):
    '''不是灯条，跳过到下一张'''
    if event.key in 'n':
        global image_files, root_path, num_img
        num_img += 1
        print('at image %s' % (num_img))
        image = cv2.imread(os.path.join(
            root_path, 'image', image_files[num_img]))
        change_image()
        return


def on_key_press_y(event):
    '''是灯条，删除'''
    if event.key in 'y':
        global image_files, root_path, num_img
        remove_light_bar(image_files, root_path, num_img)
        num_img += 1
        print('at image %s' % (num_img))
        change_image()
        return


'''上面这几个函数都是用于gui人工选图'''


def main_manual():
    image_files = []
    root_path = ''
    num_img = 0
    removed_anno = {}
    main_root_path = '/mnt/c/Users/51284/Desktop'  # change to your own path
    regions = [x for x in os.listdir(main_root_path) if (
        os.path.isdir(os.path.join(main_root_path, x))and x != 'lightbar')]
    if not os.path.exists(os.path.join(main_root_path, 'lightbar')):
        os.makedirs(os.path.join(main_root_path, 'lightbar'))
    try:
        for region in regions:
            root_path = os.path.join(main_root_path, region)
            num_img = 0
            image_files = os.listdir(os.path.join(root_path, 'image'))
            image = cv2.imread(os.path.join(
                root_path, 'image', image_files[num_img]))
            fig, ax = plt.subplots(1)
            img = ax.imshow(image)
            ax.set_title('''
                        Is there only light bar in the image?
                        press 'y' for yes,
                        press 'n' for no,
                        click to the previous image
                        press 'y' to start
                        ''')
            fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
            fig.canvas.mpl_connect('button_press_event', onclick)
            fig.canvas.mpl_connect('key_press_event', on_key_press_n)
            fig.canvas.mpl_connect('key_press_event', on_key_press_y)
            plt.show()
    except:
        pass
    finally:
        with open(os.path.join(main_root_path, 'lightbar', 'removed_anno.json'), 'w') as anno_file:
            json.dump(removed_anno, anno_file, ensure_ascii=False)
            anno_file.close()

