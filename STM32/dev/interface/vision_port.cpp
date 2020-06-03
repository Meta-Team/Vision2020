//
// Created by Kerui Zhu on 7/4/2019.
//

#include "vision_port.h"
#include "CRC16.h"
#include "CRC8.h"
#include "shell.h"
#include "memstreams.h"
#include "string.h"
#include "led.h"
#include "gimbal_logic.h"
#include "gimbal_scheduler.h"



uint8_t rx_buffer[2];
uint8_t rx_buffer2[4];
VisionPort::enemy_info_t VisionPort::enemy_info;


const UARTConfig VisionPort::UART_CONFIG = {
        nullptr,
        nullptr,
        VisionPort::uart_rx_callback,  // callback function when the buffer is filled
        nullptr,
        nullptr,
        19200, // speed
        0,
        0,
        0
};

void VisionPort::init() {

    // Start UART driver
    uartStart(UART_DRIVER, &UART_CONFIG);
    uartStartReceive(UART_DRIVER, FRAME_SOF_SIZE, rx_buffer);
    enemy_info.yaw_angle = 0;
    enemy_info.pitch_angle = 0;
}

void VisionPort::uart_rx_callback(UARTDriver *uartp) {

    chSysLockFromISR();  /// --- ENTER I-Locked state. DO NOT use LOG, printf, non I-Class functions or return ---

    // vision system sends data througn uart: [a1,a2,b1,b2,EOF]
    // (a1*100+a2)/10000*180 - 90 = yaw_diff which is the relative position of the target angle between -90 deg to +90deg; 
    // (b1*100+b2)/10000*180 - 90 = pitch_diff which is the relative position of the target angle between -90 deg to +90deg; 
    // positive direction of vision inputs: up and right
    // EOF should be 0xFE which is the end of the data package.;

    if(rx_buffer[0] == 0xFE){
        uint32_t yaw_int = rx_buffer2[0]*100+rx_buffer[1];
        uint32_t pitch_int = rx_buffer2[2]*100+rx_buffer[3];
        float yaw_diff = (float)yaw_int / 10000.0 * 180.0 - 90.0 ;
        enemy_info.yaw_angle = -yaw_diff;
        float pitch_diff = (float)pitch_int / 10000.0 * 180.0 -90.0 ;
        enemy_info.pitch_angle = -pitch_diff;
    }
    // store received data into  rx_buffer2
    else if (rx_buffer[0] == 0xFD){
        enemy_info.yaw_angle = 0;
        enemy_info.pitch_angle = 0;
    }
    else{       
        rx_buffer2[0] = rx_buffer2[1];
        rx_buffer2[1] = rx_buffer2[2];
        rx_buffer2[2] = rx_buffer2[3];
        rx_buffer2[3] = rx_buffer[0];
    }

    uartStartReceiveI(uartp, FRAME_SOF_SIZE, rx_buffer);    // receive next byte;
    chSysUnlockFromISR();  /// --- EXIT S-Locked state ---
}
