/*******************************************************************************
 * FcosFace v2 — Live Camera Demo  (main_v5.c)
 * MAX78000 Feather + OV7725 camera
 * Synthesis: fcosface_v2
 *
 * Based on main_v2.c (confirmed working: NMS, 115200 baud, face tracking).
 * Added: 112x112 grayscale thumbnail streamed as hex text after each frame.
 *
 * Serial protocol:
 *   === FRAME N  lines=224 ===
 *   [BRT] R=... G=... B=...
 *   obj raw  min=...  max=... -> sig(min)=...%  sig(max)=...%
 *   DET 0 score% x1 y1 x2 y2
 *   DET NONE
 *   FRAME_END
 *   IMG_START
 *   <112 lines of 224 hex chars>
 *   IMG_END
 *   [TIME] inf=X us  total=Y us
 *
 * CRITICAL: NO printf INSIDE THE LINE CAPTURE LOOP.
 * CRITICAL: Thumbnail must be hex text, NOT raw binary.
 ******************************************************************************/

/* ---- Active mode ---- */
// #define KAT_TEST
// #define STATIC_TEST

#define CAM_PRESCALER    0x1
#define SCORE_THRESH_F   0.55f   /* Post-calib: sigmoid(0)=50% is background; 55% = meaningful delta above it */
#define NMS_IOU_THRESH   0.30f
#define MIN_BOX_PX       30
#define MAX_DETS         64
#define MAX_DET_PRINT    5
#define CAMERA_FREQ      8330000
#define THUMB_W          112
#define THUMB_H          112

/* ---- Includes ---- */
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "mxc.h"
#include "mxc_device.h"
#include "mxc_delay.h"
#include "mxc_sys.h"
#include "uart.h"
#include "led.h"
#include "board.h"
#include "camera.h"
#include "dma.h"
#include "cnn.h"

#ifdef KAT_TEST
#include "sampledata.h"
#endif
#ifdef STATIC_TEST
#include "sample_face.h"
#endif

/* ---- Constants ---- */
#define IMAGE_W     224
#define IMAGE_H     224
#define STRIDE      8
#define GRID_W      (IMAGE_W / STRIDE)   /* 28 */
#define GRID_H      (IMAGE_H / STRIDE)   /* 28 */
#define NUM_CELLS   (GRID_W * GRID_H)    /* 784 */
#define CON_BAUD    115200

/* ---- Detection struct ---- */
typedef struct {
    float score;
    int   x1, y1, x2, y2;
    int   suppressed;
} Det;

/* ---- Globals ---- */
volatile uint32_t cnn_time;
static int32_t ml_data[CNN_NUM_OUTPUTS];
static Det dets[MAX_DETS];
static uint8_t thumb[THUMB_H][THUMB_W];

/* ---- Per-cell background calibration ----
 * CALIB_FRAMES blank-scene frames are averaged into bg_raw[].
 * During decode, bg_raw[idx] is subtracted from ml_data[idx] (raw Q14)
 * before thresholding, removing the structural top-row prior.
 * Point camera at a blank wall for the first CALIB_FRAMES frames (~30s). */
#define CALIB_FRAMES     10
static int32_t  bg_sum[NUM_CELLS];   /* accumulator during calibration */
static int32_t  bg_raw[NUM_CELLS];   /* per-cell background (mean raw Q14) */
static int       calib_done  = 0;
static int       calib_count = 0;

/* ---- Helpers ---- */
static float sigmoid_f(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static inline void fifo_write(uint32_t word)
{
    while ((*((volatile uint32_t *) 0x50000004) & 1) != 0) {}
    *((volatile uint32_t *) 0x50000008) = word & 0x00FFFFFF;
}

/* Intersection-over-Union of two boxes */
static float iou(const Det *a, const Det *b)
{
    int ix1 = a->x1 > b->x1 ? a->x1 : b->x1;
    int iy1 = a->y1 > b->y1 ? a->y1 : b->y1;
    int ix2 = a->x2 < b->x2 ? a->x2 : b->x2;
    int iy2 = a->y2 < b->y2 ? a->y2 : b->y2;
    if (ix2 <= ix1 || iy2 <= iy1) return 0.0f;
    float inter = (float)(ix2 - ix1) * (float)(iy2 - iy1);
    float area_a = (float)(a->x2 - a->x1) * (float)(a->y2 - a->y1);
    float area_b = (float)(b->x2 - b->x1) * (float)(b->y2 - b->y1);
    return inter / (area_a + area_b - inter);
}

/* send_thumbnail — stream 112x112 grayscale as hex text over UART.
 * Each row: 112 bytes as 224 hex chars + newline = 225 chars/row.
 * Total: 112 * 225 = 25,200 chars.
 * At 115200 baud (11520 chars/sec) this takes ~2.19 seconds.
 * MUST use hex text — raw binary corrupts the text protocol. */
static void send_thumbnail(void)
{
    static const char hex[] = "0123456789abcdef";
    printf("IMG_START\n");
    for (int r = 0; r < THUMB_H; r++) {
        for (int c = 0; c < THUMB_W; c++) {
            uint8_t v = thumb[r][c];
            putchar(hex[v >> 4]);
            putchar(hex[v & 0xF]);
        }
        putchar('\n');
    }
    printf("IMG_END\n");
}

/*
 * decode_nms_print
 *
 * If calibration is in progress: accumulate objectness into bg_sum, print
 * CALIB status line, return.  After CALIB_FRAMES frames: freeze bg_raw[].
 *
 * During normal operation: subtract bg_raw[idx] from each cell's raw
 * objectness before sigmoid.  This removes the structural top-row bias
 * baked into the model weights, leaving only content-driven signal.
 *
 * Then: threshold, regression decode, NMS, print.
 */
static void decode_nms_print(void)
{
    const float scale = 1.0f / 16384.0f;
    int n_raw = 0;

    /* -- Objectness diagnostic -- */
    int32_t obj_min = ml_data[0], obj_max = ml_data[0];
    for (int i = 1; i < NUM_CELLS; i++) {
        if (ml_data[i] < obj_min) { obj_min = ml_data[i]; }
        if (ml_data[i] > obj_max) { obj_max = ml_data[i]; }
    }
    printf("obj raw  min=%ld  max=%ld  -> sig(min)=%d%%  sig(max)=%d%%\n",
           (long)obj_min, (long)obj_max,
           (int)(sigmoid_f((float)obj_min * scale) * 100.0f),
           (int)(sigmoid_f((float)obj_max * scale) * 100.0f));

    /* -- Calibration phase -- */
    if (!calib_done) {
        calib_count++;
        for (int i = 0; i < NUM_CELLS; i++) {
            bg_sum[i] += ml_data[0 * NUM_CELLS + i];
        }
        printf("CALIB %d/%d  Point camera at blank wall\n",
               calib_count, CALIB_FRAMES);
        printf("DET NONE\nFRAME_END\n");
        if (calib_count >= CALIB_FRAMES) {
            for (int i = 0; i < NUM_CELLS; i++) {
                bg_raw[i] = bg_sum[i] / CALIB_FRAMES;
            }
            calib_done = 1;
            printf("CALIB DONE\n");
        }
        return;
    }

    /* -- Step 1: decode with background subtraction -- */
    for (int row = 0; row < GRID_H && n_raw < MAX_DETS; row++) {
        for (int col = 0; col < GRID_W && n_raw < MAX_DETS; col++) {
            int idx = row * GRID_W + col;

            /* Subtract per-cell background in raw Q14 space */
            int32_t raw_obj = ml_data[0 * NUM_CELLS + idx] - bg_raw[idx];
            float score = sigmoid_f((float)raw_obj * scale);
            if (score < SCORE_THRESH_F) { continue; }

            float cx = (col + 0.5f) * (float)STRIDE;
            float cy = (row + 0.5f) * (float)STRIDE;

            float rl = (float)ml_data[1 * NUM_CELLS + idx] * scale;
            float rt = (float)ml_data[2 * NUM_CELLS + idx] * scale;
            float rr = (float)ml_data[3 * NUM_CELLS + idx] * scale;
            float rb = (float)ml_data[4 * NUM_CELLS + idx] * scale;
            if (rl < -8.0f) { rl = -8.0f; } else if (rl > 8.0f) { rl = 8.0f; }
            if (rt < -8.0f) { rt = -8.0f; } else if (rt > 8.0f) { rt = 8.0f; }
            if (rr < -8.0f) { rr = -8.0f; } else if (rr > 8.0f) { rr = 8.0f; }
            if (rb < -8.0f) { rb = -8.0f; } else if (rb > 8.0f) { rb = 8.0f; }

            float x1f = cx - expf(rl) * (float)STRIDE;
            float y1f = cy - expf(rt) * (float)STRIDE;
            float x2f = cx + expf(rr) * (float)STRIDE;
            float y2f = cy + expf(rb) * (float)STRIDE;

            if (x1f < 0.0f) { x1f = 0.0f; }
            if (y1f < 0.0f) { y1f = 0.0f; }
            if (x2f > (float)IMAGE_W) { x2f = (float)IMAGE_W; }
            if (y2f > (float)IMAGE_H) { y2f = (float)IMAGE_H; }

            int x1 = (int)x1f, y1 = (int)y1f;
            int x2 = (int)x2f, y2 = (int)y2f;
            int w  = x2 - x1,  h  = y2 - y1;

            if (w <= 0 || h <= 0) { continue; }
            if (w < MIN_BOX_PX || h < MIN_BOX_PX) { continue; }

            dets[n_raw].score      = score;
            dets[n_raw].x1         = x1;
            dets[n_raw].y1         = y1;
            dets[n_raw].x2         = x2;
            dets[n_raw].y2         = y2;
            dets[n_raw].suppressed = 0;
            n_raw++;
        }
    }

    /* -- Step 2: insertion sort by score descending -- */
    for (int i = 1; i < n_raw; i++) {
        Det tmp = dets[i];
        int j = i - 1;
        while (j >= 0 && dets[j].score < tmp.score) {
            dets[j + 1] = dets[j];
            j--;
        }
        dets[j + 1] = tmp;
    }

    /* -- Step 3: greedy NMS -- */
    for (int i = 0; i < n_raw; i++) {
        if (dets[i].suppressed) { continue; }
        for (int j = i + 1; j < n_raw; j++) {
            if (dets[j].suppressed) { continue; }
            if (iou(&dets[i], &dets[j]) > NMS_IOU_THRESH) {
                dets[j].suppressed = 1;
            }
        }
    }

    /* -- Step 4: print -- */
    int n_out = 0;
    for (int i = 0; i < n_raw && n_out < MAX_DET_PRINT; i++) {
        if (dets[i].suppressed) { continue; }
        printf("DET %d %d %d %d %d %d\n",
               n_out,
               (int)(dets[i].score * 100.0f),
               dets[i].x1, dets[i].y1,
               dets[i].x2, dets[i].y2);
        n_out++;
    }
    if (n_out == 0) { printf("DET NONE\n"); }
    printf("FRAME_END\n");
}

/* ===========================================================================
 * main
 * =========================================================================*/
int main(void)
{
    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    mxc_uart_regs_t *con = MXC_UART_GET_UART(CONSOLE_UART);
    MXC_UART_Init(con, CON_BAUD, MXC_UART_IBRO_CLK);

    printf("\n[A] FcosFace v2 boot (main_v2 — NMS enabled)\n");
    MXC_Delay(SEC(2));

    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    cnn_configure();
    printf("[B] CNN ready\n");

/* ---- KAT_TEST ---- */
#ifdef KAT_TEST
    {
        static const uint32_t kat_input[] = SAMPLE_INPUT_0;
        int kat_len = (int)(sizeof(kat_input) / sizeof(kat_input[0]));
        printf("[KAT] Feeding %d words...\n", kat_len);
        cnn_start();
        for (int i = 0; i < kat_len; i++) fifo_write(kat_input[i]);
        while (cnn_time == 0) MXC_LP_EnterSleepMode();
        cnn_unload((uint32_t *)ml_data);
        cnn_stop();
        printf("[KAT] inference=%u us  ml_data[0]=%ld\n", cnn_time, (long)ml_data[0]);
        decode_nms_print();
        printf("[KAT] Done. Halting.\n");
        while (1) {}
    }

/* ---- STATIC_TEST ---- */
#elif defined(STATIC_TEST)
    printf("[STATIC] Feeding %d words...\n", SAMPLE_FACE_COUNT);
    cnn_start();
    for (int i = 0; i < SAMPLE_FACE_COUNT; i++) fifo_write(sample_face_data[i]);
    while (cnn_time == 0) MXC_LP_EnterSleepMode();
    cnn_unload((uint32_t *)ml_data);
    cnn_stop();
    printf("[STATIC] inference=%u us\n", cnn_time);
    decode_nms_print();
    printf("[STATIC] Done. Halting.\n");
    while (1) {}

/* ---- LIVE CAMERA ---- */
#else
    {
        int ret, dma_channel;

        MXC_DMA_Init();
        dma_channel = MXC_DMA_AcquireChannel();

        camera_init(CAMERA_FREQ);
        ret = camera_setup(IMAGE_W, IMAGE_H, PIXFORMAT_RGB565,
                           FIFO_FOUR_BYTE, STREAMING_DMA, dma_channel);
        if (ret != STATUS_OK) {
            printf("[ERR] camera_setup failed: %d\n", ret);
            while (1) {}
        }

        camera_write_reg(0x11, CAM_PRESCALER);
        camera_write_reg(0x13, 0xE7);   /* COM8: AEC + AGC + AWB */
        camera_write_reg(0x14, 0x48);   /* COM9: max AGC gain 32x */

        printf("[C] Camera ready  prescaler=0x%X\n", CAM_PRESCALER);
        printf("[C] Waiting 3 s for AEC...\n");
        MXC_Delay(SEC(3));
        printf("[C] Live capture started\n");

        int frame = 0;
        while (1) {
            uint8_t *data = NULL;
            frame++;

            /* Drain stale buffers */
            while ((data = get_camera_stream_buffer()) != NULL)
                release_camera_stream_buffer();

            cnn_start();
            camera_start_capture_image();

            /* Silent accumulators during capture */
            int diag_avgR = 0, diag_avgG = 0, diag_avgB = 0;
            int lines_captured = 0;

            for (int i = 0; i < IMAGE_H; i++) {
                while ((data = get_camera_stream_buffer()) == NULL) {
                    if (camera_is_image_rcv()) break;
                }
                if (data == NULL) break;
                lines_captured++;

                /* Brightness sample at midline */
                if (i == IMAGE_H / 2) {
                    int st = (IMAGE_W / 2 - 4) * 2;
                    for (int k = 0; k < 8; k++) {
                        uint8_t b0 = data[st + k * 2];
                        uint8_t b1 = data[st + k * 2 + 1];
                        diag_avgR += b0 & 0xF8;
                        diag_avgG += (uint8_t)(((b0 << 5) | ((b1 & 0xE0) >> 3)));
                        diag_avgB += (b1 & 0x1F) << 3;
                    }
                }

                /* Build thumbnail: every other line, every other pixel.
                 * BT.601 grayscale: Y = (R*77 + G*150 + B*29) >> 8.
                 * No extra buffer needed — samples directly from DMA line. */
                if ((i & 1) == 0) {
                    int tr = i >> 1;
                    if (tr < THUMB_H) {
                        for (int j = 0; j < IMAGE_W; j += 2) {
                            int tc = j >> 1;
                            uint8_t b0 = data[j * 2];
                            uint8_t b1 = data[j * 2 + 1];
                            uint8_t r  = b0 & 0xF8;
                            uint8_t g  = (uint8_t)(((b0 << 5) | ((b1 & 0xE0) >> 3)));
                            uint8_t b  = (uint8_t)((b1 & 0x1F) << 3);
                            thumb[tr][tc] = (uint8_t)(
                                ((uint32_t)r * 77 +
                                 (uint32_t)g * 150 +
                                 (uint32_t)b * 29) >> 8);
                        }
                    }
                }

                /* Unpack RGB565 -> signed INT8 -> CNN FIFO */
                uint8_t *p = data;
                for (int j = 0; j < IMAGE_W; j++) {
                    uint8_t b0 = *p++;
                    uint8_t b1 = *p++;
                    uint8_t ur = (b0 & 0xF8) ^ 0x80;
                    uint8_t ug = (uint8_t)(((b0 << 5) | ((b1 & 0xE0) >> 3)) ^ 0x80);
                    uint8_t ub = (uint8_t)((b1 << 3) ^ 0x80);
                    fifo_write(((uint32_t)ub << 16) | ((uint32_t)ug << 8) | ur);
                }
                release_camera_stream_buffer();
            }

            while (cnn_time == 0) MXC_LP_EnterSleepMode();
            cnn_unload((uint32_t *)ml_data);
            cnn_stop();

            /* Print frame header + detections (FRAME_END inside decode_nms_print) */
            printf("\n=== FRAME %d  lines=%d ===\n", frame, lines_captured);
            printf("[BRT] R=%d G=%d B=%d\n",
                   diag_avgR / 8, diag_avgG / 8, diag_avgB / 8);
            decode_nms_print();

            /* Send thumbnail after FRAME_END */
            send_thumbnail();

            /* Timing line — inference exact, total calculated */
            uint32_t inf_us   = cnn_time;
            uint32_t uart_us  = (uint32_t)(((uint64_t)(THUMB_H * (THUMB_W * 2 + 1) + 18u)
                                             * 10u * 1000000u) / CON_BAUD);
            uint32_t total_us = 200000u + inf_us + uart_us;
            printf("[TIME] inf=%u us  total=%u us\n", inf_us, total_us);

            /* Let UART TX drain before next frame */
            MXC_Delay(MSEC(20));
        }
    }
#endif

    return 0;
}