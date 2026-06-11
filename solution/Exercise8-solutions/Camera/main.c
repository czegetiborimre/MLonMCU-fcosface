/***** Includes *****/
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "mxc.h"
#include "mxc_device.h"
#include "mxc_delay.h"

#include "uart.h"
#include "led.h"
#include "board.h"

#include "camera.h"
#include "utils.h"
#include "dma.h"

#include "mxc.h"
#include "cnn.h"

#define BUTTON


/*
If BUTTON is defined, you'll need to push PB1 to capture an image frame.  Otherwise, images
will be captured continuously.
*/
// #define BUTTON

#define CAMERA_FREQ (8330000)


#define IMAGE_XRES 64
#define IMAGE_YRES 64

#define CON_BAUD 115200 * 1

volatile uint32_t cnn_time; // Stopwatch

void fail(void)
{
  printf("\n*** FAIL ***\n\n");
  while (1);
}


void load_input(uint8_t input_0[], int size)
{
  // This function loads the sample data input
  // Data has to be encoded as 0x0bgr
  // This means that the first byte of the 32 bit integer is 0, the second represents the red byte,
  // the third the green byte and the last one the blue byte

 int8_t padded_input[size*4];

 for (int i = 0; i < size; i++)
 {
	 padded_input[i*4+0] = (int8_t)(input_0[i*3+0]-128); //blue
	 padded_input[i*4+1] = (int8_t)(input_0[i*3+1]-128); //green
	 padded_input[i*4+2] = (int8_t)(input_0[i*3+2]-128); //red
	 padded_input[i*4+3] = 0;
	 //padded_input[i*4+3] = (int8_t)(input_0[i*3+0]-128);
	 //padded_input[i*4+2] = (int8_t)(input_0[i*3+1]-128);
	 //padded_input[i*4+1] = (int8_t)(input_0[i*3+2]-128);
	 //padded_input[i*4+0] = 0;
	 //printf ("-> %d %d %d %d\n\r", padded_input[i*4+3], padded_input[i*4+2], padded_input[i*4+1], padded_input[i*4+0]);
 }

  memcpy32((uint32_t *) 0x50400000, (uint32_t *)padded_input, 4096);
}


// Classification layer:
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

void softmax_layer(void)
{
  cnn_unload((uint32_t *) ml_data);
  softmax_q17p14_q15((const q31_t *) ml_data, CNN_NUM_OUTPUTS, ml_softmax);
}

void process_img(void)
{
    uint8_t *raw;
    uint32_t imgLen;
    uint32_t w, h;
    int i;
    int digs, tens;

    // Get the details of the image from the camera driver.
    camera_get_image(&raw, &imgLen, &w, &h);

    uint8_t img[w*h*3];
    //uint8_t* img = input_0_sample;

    //Image is in format 565 (2 bytes per pixel) and needs to be parsed into the standard rgb format (3 bytes per pixel)
    utils_parse_image_rgb565(raw, (uint8_t*)img, w, h);


    printf("Waiting...\n");

    // DO NOT DELETE THIS LINE:
    MXC_Delay(SEC(2)); // Let debugger interrupt if needed

    // Enable peripheral, enable CNN interrupt, turn on CNN clock
    // CNN clock: APB (50 MHz) div 1
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);

    printf("\n*** CNN Inference Test memenet ***\n");

    cnn_init(); // Bring state machine into consistent state
    cnn_load_weights(); // Load kernels
    cnn_load_bias();
    cnn_configure(); // Configure state machine
    printf("Load input\r\n");
    load_input((uint8_t*)img, w*h); // Load data input
    printf("Run inference\r\n");
    cnn_start(); // Start CNN processing

    while (cnn_time == 0)
      MXC_LP_EnterSleepMode(); // Wait for CNN

    softmax_layer();

    printf("\n*** PASS ***\n\n");

  #ifdef CNN_INFERENCE_TIMER
    printf("Approximate inference time: %u us\n\n", cnn_time);
  #endif

    cnn_disable(); // Shut down CNN clock, disable peripheral

    char result[2048];
    int meme = -1;

    printf("Classification results:\n");
    sprintf(result, "Classification results:\n");
    for (i = 0; i < CNN_NUM_OUTPUTS; i++) {
      digs = (1000 * ml_softmax[i] + 0x4000) >> 15;
      tens = digs % 10;
      digs = digs / 10;
      if (digs > 40)
    	  meme = i;
      printf("[%7d] -> Class %d: %d.%d%%\n", ml_data[i], i, digs, tens);
      sprintf(result + strlen(result), "[%7d] -> Class %d: %d.%d%%\n", ml_data[i], i, digs, tens);
    }

    switch(meme)
    {
    case 0:
    	sprintf(result + strlen(result), "This is fine\n");
    	break;
    case 1:
    	sprintf(result + strlen(result), "Fry\n");
    	break;
    case 2:
    	sprintf(result + strlen(result), "Grumpy cat\n");
    	break;
    case 3:
    	sprintf(result + strlen(result), "Pikachu\n");
    	break;
    default:
    	sprintf(result + strlen(result), "Not a meme :(\n");

    }


    //Send image and results to the pc

    // In the original format each pixel is 2 bytes: 5 bits for red, 6 for green, 5 for blue
    // In the new format each pixel is 3 bytes one for each color
    // Therefore the dimension of the new image is 3/2 times the dimension of the raw image obtained by the camera
    utils_send_img_to_pc(img, imgLen*3/2, w, h, (uint8_t*)"RGB888");
    utils_send_results_to_pc(result);
}

// *****************************************************************************
int main(void)
{
    int ret = 0;
    int slaveAddress;
    int id;
    int dma_channel;

    /* Enable cache */
    MXC_ICC_Enable(MXC_ICC0);

    /* Set system clock to 100 MHz */
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    // Initialize DMA for camera interface
    MXC_DMA_Init();
    dma_channel = MXC_DMA_AcquireChannel();

    mxc_uart_regs_t *ConsoleUart = MXC_UART_GET_UART(CONSOLE_UART);

    if ((ret = MXC_UART_Init(ConsoleUart, CON_BAUD, MXC_UART_IBRO_CLK)) != E_NO_ERROR) {
        return ret;
    }

    // Initialize the camera driver.
    camera_init(CAMERA_FREQ);
    printf("\n\nCamera initialized\n");

    slaveAddress = camera_get_slave_address();
    printf("Camera I2C slave address: %02x\n", slaveAddress);

    // Obtain the manufacturer ID of the camera.
    ret = camera_get_manufacture_id(&id);

    if (ret != STATUS_OK) {
        printf("Error returned from reading camera id. Error %d\n", ret);
        return -1;
    }

    printf("Camera ID detected: %04x\n", id);


    ret = camera_setup(IMAGE_XRES, IMAGE_YRES, PIXFORMAT_RGB565, FIFO_FOUR_BYTE, USE_DMA,
                       dma_channel); // RGB565



    if (ret != STATUS_OK) {
        printf("Error returned from setting up camera. Error %d\n", ret);
        return -1;
    }

    MXC_Delay(SEC(1));

    // Start capturing a first camera image frame.
    printf("Starting\n");
#ifdef BUTTON
    while (!PB_Get(0)) {}
#endif

    camera_start_capture_image();

    while (1) {
        // Check if image is acquired.
        if (camera_is_image_rcv())
        {
            
            // Process the image, send it through the UART console.
            process_img();

            // Prepare for another frame capture.
            LED_Toggle(LED_GREEN);
#ifdef BUTTON
            while (!PB_Get(0)) {}
#endif
            camera_start_capture_image();
        }
    }

    return ret;
}
