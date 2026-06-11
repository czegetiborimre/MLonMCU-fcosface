/******************************************************************************
 * facedet_tinierssd — UART input mode
 * CNN is set up ONCE; only cnn_start() runs per frame (ADI reference pattern).
 *
 * Protocol:
 *   MCU -> PC : "WAITING\n"
 *   PC  -> MCU: IMAGE_SIZE_X * IMAGE_SIZE_Y * 2 bytes, big-endian RGB565
 *               (byte0 = RRRRRGGG, byte1 = GGGBBBBB)
 *   MCU -> PC : "SCORE:bg:face:sm:thresh\n"   (diagnostic from post_process)
 *   MCU -> PC : "DETECT:0\n"  or  "DETECT:1\n"
 *   MCU -> PC : "BOX:x1:y1:x2:y2\n"           (only on DETECT:1)
 ******************************************************************************/

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "mxc_errors.h"
#include "post_process.h"
#include "led.h"
#include "board.h"

volatile uint32_t cnn_time;
extern volatile uint8_t face_detected;
extern uint8_t box[4];

void fail(void)
{
    printf("\n*** FAIL ***\n\n");
    while (1) {}
}

static inline uint8_t uart_getc(void)
{
    int c;
    while ((c = MXC_UART_ReadCharacterRaw(MXC_UART0)) < 0) {}
    return (uint8_t)c;
}

/* Read IMAGE_SIZE_X*IMAGE_SIZE_Y RGB565-BE pixels from UART, feed CNN FIFO 0. */
void load_input(void)
{
    const int total = IMAGE_SIZE_X * IMAGE_SIZE_Y;  /* 37632 pixels */
    uint8_t hi, lo;

    for (int i = 0; i < total; i++) {
        hi = uart_getc();   /* RRRRRGGG */
        lo = uart_getc();   /* GGGBBBBB */

        uint8_t r5 = (hi >> 3) & 0x1F;
        uint8_t g6 = ((hi & 0x07) << 3) | ((lo >> 5) & 0x07);
        uint8_t b5 = lo & 0x1F;

        uint8_t ur = r5 << 3;
        uint8_t ug = g6 << 2;
        uint8_t ub = b5 << 3;

        /* Normalize [0,255] -> [-128,127] (same as cam02 demo) */
        int8_t r = (int8_t)(ur - 128);
        int8_t g = (int8_t)(ug - 128);
        int8_t b = (int8_t)(ub - 128);

        uint32_t rgb888 = (uint32_t)(uint8_t)r
                        | ((uint32_t)(uint8_t)g << 8)
                        | ((uint32_t)(uint8_t)b << 16);

        /* Wait for FIFO 0 space, then write */
        while ((*((volatile uint32_t *)0x50000004) & 1) != 0) {}
        *((volatile uint32_t *)0x50000008) = rgb888;
    }
}

int main(void)
{
    MXC_ICC_Enable(MXC_ICC0);

    /* 100 MHz */
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    /* Force UART0 to 921600 regardless of CONSOLE_BAUD build flag */
    MXC_UART_Shutdown(MXC_UART0);
    MXC_UART_Init(MXC_UART0, 921600, MXC_UART_APB_CLK);

    /* Let debugger attach */
    MXC_Delay(SEC(2));

    printf("\n*** facedet_tinierssd UART mode ***\n");
    printf("Frame size: %d bytes\n", IMAGE_SIZE_X * IMAGE_SIZE_Y * 2);

    /* ── CNN setup: ONCE.  Matches ADI's reference FIFO streaming pattern.
     *    Calling cnn_init() / cnn_configure() per-frame can leave the
     *    accelerator half-armed: it consumes FIFO bytes but never re-runs
     *    inference, so the output SRAM stays frozen on frame 1. */
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK,
               MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    cnn_configure();
    printf("CNN ready.\n");

    while (1) {
        face_detected = 0;
        LED_Off(0);
        LED_Off(1);

        /* Per-frame: arm the CNN, announce ready, feed the FIFO. */
        cnn_time = 0;
        cnn_start();

        printf("WAITING\n");
        LED_On(1);

        load_input();   /* ~1.17 s at 921600 baud for 75264 bytes */

        while (cnn_time == 0) {
            MXC_LP_EnterSleepMode();
        }
        printf("INFER:%u\n", (unsigned)cnn_time);

        get_priors();
        localize_objects();

        LED_Off(1);

        if (face_detected) {
            LED_On(0);
            printf("DETECT:1\n");
            printf("BOX:%d:%d:%d:%d\n", box[0], box[1], box[2], box[3]);
        } else {
            printf("DETECT:0\n");
        }
    }

    return 0;
}