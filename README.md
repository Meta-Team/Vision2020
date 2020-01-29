# Meta-Vision2020

ZJU-UIUC Meta 战队视觉组嵌入式程序工程，基于NVIDIA Jetson Nano，使用C++

工程将会包含步兵、哨兵机器人的自动化火控系统，其中哨兵主要是对敌方机器人的识别检测，步兵还有对大风车Rune的识别检测与预判。目前正在完善对机器人装甲板的识别系统。

# TODOs
- [x] cmake的编译链
    - [x] v4l2的库依赖问题 (rebuild OpenCV with v4l2 and link v4l2 library)
- [ ] 摄像头参数相适应
    - 用~v4l2~还是DJI的RMVideoCapture
    - [ ] 上机测试
- [ ] 灯条识别的方式修改
    - [ ] 数字模板
    - [ ] 上机测试
- [ ] 自己车的串口通信
    - [ ] 传回一个最优目标
    - [ ] 传回多个可选目标 (带装甲板信息)
- [ ] 大符识别
- [ ] 目标预测
- [ ] 优化目标装甲板筛选

# STRUCTURE

update: 20/1/27

emmmm