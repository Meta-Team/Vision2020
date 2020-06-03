import serial
import serial.tools.list_ports
import time
import math

port_list = list(serial.tools.list_ports.comports())
use_usb = 1


dev = "/dev/ttyTHS1"
for i in port_list:
    print(i)
if len(port_list) > 0 and use_usb:
    dev = list(port_list[0])[0]
print('using:',dev)
    
# try to open UART
try:
    ser = serial.Serial(dev, 19200, timeout=10)
    have_uart = 1
except:
    print("open UART error!")
    
    
def uart_send(arr):
    ser_data = bytes([63-arr[0],arr[1],0xFF,arr[2] // 100,arr[2] % 100,arr[3] // 100 , arr[3] % 100, 0xFE ])
    if have_uart:
        try:
            ser.write(ser_data)
            print("ser write:", ser_data)
        except:
            print('send UART error!')

while 1:
   
    for i in range(50):     # up
        uart_send([i,i,5000,5200])
    time.sleep(0.02)

    for i in range(50):     # right
        uart_send([i,i,5200,5000])
    time.sleep(0.02)
    
    for i in range(50):     # down
        uart_send([i,i,5000,4800])
        
    time.sleep(0.02)

    for i in range(50):     # left
        uart_send([i,i,4800,5000])
        
    time.sleep(0.02)
   
    #ser.write(bytes([0xff,0x00]))
    #time.sleep(1)