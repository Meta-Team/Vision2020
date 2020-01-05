# -*- coding: utf-8 -*- 
# %%
# import pretty_errors
import xml.etree.ElementTree as ET
import os
import json
import xmltodict
import cv2
import numpy

root_path = "/mnt/e/robomaster/DJI ROCO"  # change to your own path
dump_path = os.path.join(os.path.dirname(root_path), 'mydump')
regions = [x for x in os.listdir(root_path)
           if os.path.isdir(os.path.join(root_path, x))]
if not os.path.exists(dump_path):
    os.makedirs(dump_path)
# %%

def main():
    for region in regions:
        # for each folder in official dataset, make corresponding dump folders
        os.chdir(os.path.join(root_path, region))
        print("working on %s" % region)

        image_path = os.path.join(root_path, region, "image")
        annotation_path = os.path.join(root_path, region, "image_annotation")

        image_dump_path = os.path.join(dump_path, region, "image")
        annotation_dump_path = os.path.join(
            dump_path, region, "image_annotation") 
        if not os.path.exists(image_dump_path):
            os.makedirs(image_dump_path)
        if not os.path.exists(annotation_dump_path):
            os.makedirs(annotation_dump_path)

        # for each annotation file, find the armor information
        # and dump as json file
        # also cut the useful part of image out
        for file in os.listdir(annotation_path):
            num_object = 0 # count the number of armors
            anno_refine_json = {} # dictionary of refined armor info

            # skip if already refined 
            json_name = os.path.splitext(file)[0]+'.json'
            if os.path.exists(os.path.join(annotation_dump_path, json_name)):
                continue
            
            # skip if no object in annotation
            print("refining annotation: "+file)
            with open(os.path.join(annotation_path, file), 'r') as xml_file:
                xml_str = xml_file.read()
                ann_dict = xmltodict.parse(xml_str)
                xml_file.close()
            if not 'object' in ann_dict['annotation']:
                continue

            # 这里用file是无奈之举，因为annotation中用英文名，而img_src是中文名，只能用annotation的文件名代替了
            img_src = cv2.imread(os.path.join(
                image_path, os.path.splitext(file)[0]+'.jpg'))

            # 一张图片一个json文件
            for object_ in ann_dict['annotation']["object"]:
                # annotation只有一个object的情况
                if 'name' in ann_dict['annotation']["object"]:
                    object_ = ann_dict['annotation']["object"]

                num_object += 1 #文件名从1开始计数
                img_name = os.path.splitext(file)[0] + "_%s" % num_object + ".jpg"
                if object_['name'] == 'armor':
                    # cut image (corespondedly)
                    ymin = int(float(object_['bndbox']['ymin']))
                    ymax = int(float(object_['bndbox']['ymax']))
                    xmin = int(float(object_['bndbox']['xmin']))
                    xmax = int(float(object_['bndbox']['xmax']))
                    try:
                        # skip if connot find img_src (NoneType for img_src)
                        img = img_src[ymin:ymax, xmin:xmax]
                        cv2.imwrite(os.path.join(image_dump_path, img_name), img)
                        # collect useful information
                        anno_refine_json[img_name] = [object_['armor_class'], object_['armor_color']]
                    except TypeError as e:
                        print(e, 'on file: ' +ann_dict['annotation']['filename'])
                        print("中英文名都没有对应的image")
                        break
            # dump to json
            with open(os.path.join(annotation_dump_path, json_name), 'w') as json_file:
                json.dump(anno_refine_json, json_file, ensure_ascii=False)
                json_file.close()
    return

# 用xml的elementtree库来做分析，最后没有用到
def test_with_xml_tree():
    for region in regions:
        os.chdir(region)
        print("working on %s" % region)

        # image_path = os.path.join(region, "image")
        annotation_path = os.path.join(region, "image_annotation")

        # for each annotation file, find the arnor information
        # and dump as json file
        # also cut the useful part of image out
        for file in os.listdir(annotation_path):
            print("refining annotation: "+file)
            xml_tree = ET.parse(os.path.join(annotation_path, file))
            root = xml_tree.getroot()
            for object_ in root.findall('object'):
                print("find an object")

                if object_.find('name').text == "armor":
                    for child in object_:
                        print(child)
                        # collect useful information
                        # dump to json
                        # cut image (corespondedly)
                else:
                    print('not armor object')


            break
        break
    return

if __name__ == "__main__":
    main()



# %%
'''
xml to dict后的结构
OrderedDict(
    [
        ('annotation', OrderedDict([
            ('filename', 'CUBOTVsNEXT E_BO2_2_47.jpg'),
            ('version', 'v3.0'),
            ('review_status', 'passed'),
            ('time_elapsed', '139'),
            ('img_label', None),
            ('size', OrderedDict(
                [('width', '1920'),
                 ('height', '1080')])),
            ('padding_pxt', OrderedDict(
                [('padding_x', '76'),
                 ('padding_y', '43')])),
            ('object',
             [OrderedDict(
                 [('name', 'armor'),
                  ('armor_class', '3'),
                  ('armor_color', 'red'),
                  ('generated', '0'),
                  ('is_incorrect', '0'),
                  (''bndbox'',' Ord'eredDict(
                      [('xmin', '359.983'),
                       ('ymin', '925.287'),
                       ('xmax', '391.667'),
                       ('ymax', '943.03')]))]),
              OrderedDict(
                  [('name', 'armor'),
                   ('armor_class', '4'),
                   ('armor_color', 'red'),
                   ('generated', '0'),
                   ('is_incorrect', '0'),
                   (''bndbox'',' Ord'eredDict(
                       [('xmin', '539.952'),
                        ('ymin', '912.613'),
                        ('xmax', '574.172'),
                        ('ymax', '931.624')]))]),
              OrderedDict([('name', 'armor'), ('armor_class', '4'), ('armor_color', 'red'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '514.604'), ('ymin', '888.533'), ('xmax', '538.685'), ('ymax', '907.543')]))]), OrderedDict([('name', 'armor'), ('armor_class', '1'), ('armor_color', 'blue'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1121.68'), ('ymin', '256.104'), ('xmax', '1148.3'), ('ymax', '270.046')]))]), OrderedDict([('name', 'armor'), ('armor_class', '2'), ('armor_color', 'blue'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '302.95'), ('ymin', '545.07'), ('xmax', '329.565'), ('ymax', '562.813')]))]), OrderedDict([('name', 'armor'), ('armor_class', '4'), ('armor_color', 'blue'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1249.69'), ('ymin', '988.657'), ('xmax', '1271.24'), ('ymax', '1008.93')]))]), OrderedDict([('name', 'armor'), ('armor_class', '4'), ('armor_color', 'blue'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1195.19'), ('ymin', '979.785'), ('xmax', '1221.81'), ('ymax', '1000.06')]))]), OrderedDict([('name', 'armor'), ('armor_class', '4'), ('armor_color', 'blue'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1196.46'), ('ymin', '953.17'), ('xmax', '1219.27'), ('ymax', '965.843')]))]), OrderedDict([('name', 'armor'), ('armor_class', '6'), ('armor_color', 'blue'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1333.34'), ('ymin', '239.628'), ('xmax', '1344.75'), ('ymax', '294.126')]))]), OrderedDict([('name', 'armor'), ('armor_class', '6'), ('armor_color', 'blue'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1415.72'), ('ymin', '213.013'), ('xmax', '1429.66'), ('ymax', '256.104')]))]), OrderedDict([('name', 'armor'), ('armor_class', '6'), ('armor_color', 'blue'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1441.07'), ('ymin', '223.152'), ('xmax', '1455.01'), ('ymax', '262.441')]))]), OrderedDict([('name', 'armor'), ('armor_class', '6'), ('armor_color', 'blue'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1484.16'), ('ymin', '251.035'), ('xmax', '1495.57'), ('ymax', '305.533')]))]), OrderedDict([('name', 'armor'), ('armor_class', '8'), ('armor_color', 'blue'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1399.24'), ('ymin', '262.441'), ('xmax', '1434.73'), ('ymax', '277.65')]))]), OrderedDict([('name', 'armor'), ('armor_class', '8'), ('armor_color', 'blue'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1403.05'), ('ymin', '196.537'), ('xmax', '1441.07'), ('ymax', '209.211')]))]), OrderedDict([('name', 'armor'), ('armor_class', '7'), ('armor_color', 'grey'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1152.1'), ('ymin', '277.65'), ('xmax', '1181.25'), ('ymax', '292.859')]))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         ]), OrderedDict([('name', 'armor'), ('armor_class', '3'), ('armor_color', 'blue'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1768.05'), ('ymin', '731.376'), ('xmax', '1789.6'), ('ymax', '749.12')]))]), OrderedDict([('name', 'armor'), ('armor_class', '3'), ('armor_color', 'blue'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1718.63'), ('ymin', '737.713'), ('xmax', '1750.31'), ('ymax', '756.724')]))]), OrderedDict([('name', 'base'), ('difficulty', '0'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1327'), ('ymin', '188.933'), ('xmax', '1499.37'), ('ymax', '318.207')]))]), OrderedDict([('name', 'watcher'), ('difficulty', '0'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1144.5'), ('ymin', '253.57'), ('xmax', '1211.67'), ('ymax', '311.87')]))]), OrderedDict([('name', 'car'), ('difficulty', '1'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '209.163'), ('ymin', '450.015'), ('xmax', '356.18'), ('ymax', '581.824')]))]), OrderedDict([('name', 'car'), ('difficulty', '1'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1106.48'), ('ymin', '201.607'), ('xmax', '1199'), ('ymax', '299.196')]))]), OrderedDict([('name', 'car'), ('difficulty', '0'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1168.58'), ('ymin', '925.287'), ('xmax', '1282.64'), ('ymax', '1027.95')]))]), OrderedDict([('name', 'car'), ('difficulty', '0'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '1693.28'), ('ymin', '689.552'), ('xmax', '1802.27'), ('ymax', '773.2')]))]), OrderedDict([('name', 'car'), ('difficulty', '0'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '500.663'), ('ymin', '855.58'), ('xmax', '600.787'), ('ymax', '944.298')]))]), OrderedDict([('name', 'car'), ('difficulty', '0'), ('generated', '0'), ('is_incorrect', '0'), (''bndbox'',' Ord'eredDict([('xmin', '334.635'), ('ymin', '856.848'), ('xmax', '423.352'), ('ymax', '953.17')]))])])
        ]
        )
        )
    ]
)
'''
