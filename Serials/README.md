# 通信

目前只有与哨兵进行通信的版本。需要再和电控组的同学协商后完成。
# 2020.1.30 Update
Jetson nano上没有/dev/ttyTHS2，故改用/dev/ttyTHS1。TX是nano上的PIN8，RX是PIN10。使用时连上TX,RX,GND三个脚即可。  
其他代码应该没问题，可以食用。
![avatar](https://img-blog.csdnimg.cn/20190812174551977.png)