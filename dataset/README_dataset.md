# 装甲板数据集

通过处理官方的数据集，挑选出了装甲板的区域，和相应的标签

## 文件结构：
    mydump2
    ├── robomaster_Central China Regional Competition
    │   ├── image               (装甲板图片)
    │   ├── image_annotation    (装甲板标签，按图片分类)
    │   ├── lightbar            (找出来的灯条，其中有removed_anno.json记录了灯条信息)
    │   └── merged_armor.json   (装甲板标签集合)
    ├── robomaster_Final Tournament
    │   ├── image
    │   ├── image_annotation
    │   ├── lightbar
    │   └── merged_armor.json
    ├── robomaster_North China Regional Competition
    │   ├── image
    │   ├── image_annotation
    │   ├── lightbar
    │   └── merged_armor.json
    └── robomaster_South China Regional Competition
        ├── image
        ├── image_annotation
        ├── lightbar
        └── merged_armor.json


## 工具：
    dataset
    ├── README_dataset.md (本文件)
    ├── example (未经灯条处理的装甲板数据集势力)
    │   ├── image
    │   └── image_annotation
    ├── line_test.py (仅测试用, 用于测试直线识别)
    ├── manual_test.py (仅测试用, 用于人工挑选图形界面)
    ├── merge_json.py (将image_annotation中的json合并成一个文件)
    ├── pick_armor.py (从原数据集中割出所有的装甲板图片, 并把标签信息变成json)
    └── remove_light_bar.py (移除选出装甲板中的灯条图片, 同时修改json)

## 使用：
1. 把官方数据集解压，得到DJI ROCO文件夹
2. 修改`pick_armor.py`中的`root_path`，`dump_path`，运行
3. 修改`remove_light_bar.py`中的`main_root_path`为上述`dump_path`路径，运行
4. 修改`merge_json.py`中的`main_root_path`为上述`dump_path`路径，运行