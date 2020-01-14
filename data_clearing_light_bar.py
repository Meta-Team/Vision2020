# -*- coding: utf-8 -*-
# %%
import os
import json
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import shutil


# %%
def is_light_bar(image, avg_threshold=0.3*255):
    '''判断给的图片是不是纯灯条，判断方法：1.长宽比，2.图像红蓝通道平均值'''
    if (image.shape[0] > 1.5*image.shape[1] and
            (image[:, :, 0].mean() > avg_threshold or image[:, :, 2].mean() > avg_threshold)):
        return True
    return False


# img_open = Image.open('img_png.png')
# img_png = ImageTk.PhotoImage(img_open)
# label_img = Tkinter.Label(root, image = img_png)
# label_img.pack()

def remove_light_bar(image_files, root_path, num_img):
    '''去除图像文件夹中的图片，同时去除annotation文件中的记录，把这些挑选出来的放在其他的文件中，待审核'''
    global removed_anno
    print('remove_light_bar : %s' % (image_files[num_img]))
    image_name = image_files[num_img]
    anno_name = '_'.join(image_name.split('_')[:-1])+'.json'
    with open(os.path.join(root_path, 'image_annotation', anno_name), 'r') as anno_file:
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


def show_light_bar():
    '''用于测试，显示不同通道平均值选出的图片样本'''
    root_path = '/home/gavin/Projects/Vision2020/example/image'
    num_img = 1
    for image_file in os.listdir(root_path):
        image = cv2.imread(os.path.join(root_path, image_file))
        if is_light_bar(image):
            plt.subplot(5, 5, num_img)
            print(image_file+': is light bar '+str(num_img))
            plt.imshow(image)
            num_img += 1
        if num_img > 25:
            break
    plt.show()
    return


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

# %%


if __name__ == "__main__":
    '''筛选出是灯条的照片'''
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
                        图片中是否只有灯条？
                        如果是，按y；如果是装甲板，按n；单击图片返回上一张
                        单击y开始
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

# %%
# if __name__ == "__main__":
#     root_path = ''  # TODO change to your own path
#     regions = [x for x in os.listdir(root_path)
#                if os.path.isdir(os.path.join(root_path, x))]
#     for region in regions:
#         image_files = os.listdir(os.path.join(root_path, region, 'image'))
#         for image_file in image_files:

#             image = cv2.imread(os.path.join(
#                 root_path, region, 'image', image_file))
#             if is_light_bar(image):
#                 move_light_bar(image_file, os.path.join(root_path, region))

    # for image_file in os.listdir(root_path):
    #     image = cv2.imread(os.path.join(root_path, image_file))
    #     if is_light_bar(image):
    #         plt.subplot(132)
    #         plt.imshow(image)
