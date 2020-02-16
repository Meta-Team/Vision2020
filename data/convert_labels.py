import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import pypinyin

classes = ['car', 'watcher', 'base', 'armor']  # 自己训练的类别

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open('Annotations/%s.xml' % (image_id))
    out_file = open('labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        difficulty = obj.find('difficulty')
        if cls not in classes or (difficulty != None and int(difficulty.text) > 2):
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        if b[0] < 0.01*w or b[2] < 0.01*w or b[1] > w*0.99 or b[3] > h*0.99:
            continue 
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " +
                       " ".join([str(a) for a in bb]) + '\n')

path1 = 'images/'
path2 = 'Annotations/'
all_names = [path1+i for i in os.listdir(path1)] + [path2+i for i in os.listdir(path2)]
for filename in all_names: 
    filename1 = pypinyin.pinyin(filename, style=pypinyin.FIRST_LETTER) 
    filename2 = []
    for ch in filename1:
        filename2.extend(ch)
    os.rename(filename, ''.join(filename2))


wd = getcwd()
if not os.path.exists('labels/'):
    os.makedirs('labels/')
total_xml = os.listdir('Annotations')
image_ids = [i[:-4] for i in total_xml]
for image_id in image_ids:
    convert_annotation(image_id)
