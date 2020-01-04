# 基于mxnet的数字识别模型

## ISSUE:
   - [ ] 官方的装甲板box把两边的灯条放进去了，自己的应该没有，要割掉这一部分
   - [ ] 官方识别的装甲板有一些全是长灯条，让人头疼
   - [ ] `DJI ROCO\robomaster_Central China Regional Competition\image_annotation\AllianceVsArtisans_BO2_2_0.xml这个annotation`找不到原图文件

## 2020.1.4
官方数据集的预处理计划启动：
1. 把官方的annotation中装甲板信息提取出来，做成自己的json
2. 把官方的原图中的装甲板部分分割出来（可能做数字部分的切割）
3. 把这两部分信息整合成自己的训练集，初步训练模型之后在根据自己的数据集训练
   
   把部分预处理图片上传到example中



## 2019.12.26
基于mxnet框架，用了lenet的结构搭的模型。还没处理训练集，没训练
