/******************************************************************************
 * Copyright (C) 2023 Maxim Integrated Products, Inc., All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * Except as contained in this notice, the name of Maxim Integrated
 * Products, Inc. shall not be used except as stated in the Maxim Integrated
 * Products, Inc. Branding Policy.
 *
 * The mere transfer of this software does not imply any licenses
 * of trade secrets, proprietary technology, copyrights, patents,
 * trademarks, maskwork rights, or any other form of intellectual
 * property whatsoever. Maxim Integrated Products, Inc. retains all
 * ownership rights.
 *
 ******************************************************************************/
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "mxc_device.h"
#include "led.h"
#include "board.h"
#include "mxc_delay.h"
#include "uart.h"
#include "rtc.h"
#include "utils.h"

#pragma GCC optimize("-O0")

#define DEBUG_COMPORT MXC_UART0

/***************************** VARIABLES *************************************/

/************************    PUBLIC FUNCTIONS  *******************************/
//void utils_delay_ms(uint32_t ms)
//{
//    MXC_Delay(ms * 1000UL);
//}

uint32_t utils_get_time_ms(void)
{
    int sec;
    double subsec;
    uint32_t ms;

    subsec = MXC_RTC_GetSubSecond() / 4096.0;
    sec = MXC_RTC_GetSecond();

    ms = (sec * 1000) + (int)(subsec * 1000);

    return ms;
}

static void utils_send_byte(mxc_uart_regs_t *uart, uint8_t value)
{
    while (MXC_UART_WriteCharacter(uart, value) == E_OVERFLOW) {}
}

static void utils_send_bytes(mxc_uart_regs_t *uart, uint8_t *ptr, int length)
{
    int i;

    for (i = 0; i < length; i++) {
        utils_send_byte(uart, ptr[i]);
    }
}

int utils_send_raw_img_to_pc(uint8_t *img, uint32_t imgLen, int w, int h, uint8_t *pixelformat)
{
    int len;

    // Transmit the start token
    len = 5;
    utils_send_bytes(DEBUG_COMPORT, (uint8_t *) "New image\n", 10);

    // Transmit the width of the image
    utils_send_byte(DEBUG_COMPORT, (w >> 8) & 0xff); // high byte
    utils_send_byte(DEBUG_COMPORT, (w >> 0) & 0xff); // low byte
    // Transmit the height of the image
    utils_send_byte(DEBUG_COMPORT, (h >> 8) & 0xff); // high byte
    utils_send_byte(DEBUG_COMPORT, (h >> 0) & 0xff); // low byte

    // Transmit the pixel format of the image
    len = strlen((char *)pixelformat);
    utils_send_byte(DEBUG_COMPORT, len & 0xff);
    utils_send_bytes(DEBUG_COMPORT, pixelformat, len);

    // Transmit the image length in bytes
    utils_send_byte(DEBUG_COMPORT, (imgLen >> 24) & 0xff); // high byte
    utils_send_byte(DEBUG_COMPORT, (imgLen >> 16) & 0xff); // low byte
    utils_send_byte(DEBUG_COMPORT, (imgLen >> 8) & 0xff); // low byte
    utils_send_byte(DEBUG_COMPORT, (imgLen >> 0) & 0xff); // low byte

    // Send the image pixel bytes
    while (imgLen) {
        len = imgLen;
        utils_send_bytes(DEBUG_COMPORT, img, len);
        img += len;
        imgLen -= len;
    }

    return 0;
}

int utils_send_img_to_pc(uint8_t *img, uint32_t imgLen, int w, int h, uint8_t *pixelformat)
{

    int len;

    // Transmit the start token
    len = 5;
    utils_send_bytes(DEBUG_COMPORT, (uint8_t *) "New image\n", 10);

    // Transmit the width of the image
    utils_send_byte(DEBUG_COMPORT, (w >> 8) & 0xff); // high byte
    utils_send_byte(DEBUG_COMPORT, (w >> 0) & 0xff); // low byte
    // Transmit the height of the image
    utils_send_byte(DEBUG_COMPORT, (h >> 8) & 0xff); // high byte
    utils_send_byte(DEBUG_COMPORT, (h >> 0) & 0xff); // low byte

    // Transmit the pixel format of the image
       len = strlen((char *)pixelformat);
       utils_send_byte(DEBUG_COMPORT, len & 0xff);
       utils_send_bytes(DEBUG_COMPORT, pixelformat, len);

    // Transmit the image length in bytes
    utils_send_byte(DEBUG_COMPORT, (imgLen >> 24) & 0xff); // high byte
    utils_send_byte(DEBUG_COMPORT, (imgLen >> 16) & 0xff); // low byte
    utils_send_byte(DEBUG_COMPORT, (imgLen >> 8) & 0xff); // low byte
    utils_send_byte(DEBUG_COMPORT, (imgLen >> 0) & 0xff); // low byte

    // Send the image pixel bytes
    while (imgLen) {
        len = imgLen;
        utils_send_bytes(DEBUG_COMPORT, img, len);
        img += len;
        imgLen -= len;
    }

    return 0;
}

void utils_parse_image_rgb565(uint8_t *inp, uint8_t *out, uint32_t w, uint32_t h)
{
    for (int i = 0; i < w*h*2; i+=2)
    {
        // The pixel composition is the following: bbbbbggg gggrrrrr
        // therefore we have to do some masking
        uint8_t b = inp[i+1] & 0x1f;
        uint8_t g = ((inp[i] & 0x07) << 3) | ((inp[i+1] & 0xe0) >> 5);
        uint8_t r = (inp[i] & 0xf8) >> 3;
        int index = i/2;
        out[index * 3 + 0] = (uint8_t)((r / 31.0) * 255); // Red and Blue are encoded in 5 bits
        out[index * 3 + 1] = (uint8_t)((g / 63.0) * 255); // Green has a bit more!
        out[index * 3 + 2] = (uint8_t)((b / 31.0) * 255);
    }
    return;
}

void utils_send_results_to_pc(char str[])
{
	int len = strlen(str);
	utils_send_byte(DEBUG_COMPORT, len & 0xff);
	utils_send_bytes(DEBUG_COMPORT, (uint8_t *) str, len);
}




