/******************************************************************************
 * post_process.h — TinierSSD (MAX78000 facedet)
 ******************************************************************************/

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// #define RETURN_LARGEST // If defined returns the largest face detected

#define SQUARE(x) ((x) * (x))
#define MULT(x, y) ((x) * (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define NUM_ARS     2
#define NUM_SCALES  2
#define NUM_CLASSES 3

#define LOC_DIM 4 /* (x1, y1, x2, y2) */

#define IMAGE_SIZE_X 168
#define IMAGE_SIZE_Y 224

#define NUM_PRIORS_PER_AR 623
#define NUM_PRIORS        (NUM_PRIORS_PER_AR * NUM_ARS)

#ifdef RETURN_LARGEST
#define MAX_PRIORS 20
#else
#define MAX_PRIORS 1
#endif

/* MIN_CLASS_SCORE is uint16 softmax probability (0..65535).
 * 6553  ~  10 %  — start here; raise to 16384 (~25%) once detection works
 *                  if you get false positives, or lower if a real face misses. */
#define MIN_CLASS_SCORE      19660  /*6553, 19660 ~30 %*/
#define MAX_ALLOWED_OVERLAP  0.3f

void get_priors(void);
void nms(void);
void get_cxcy(float *cxcy, int prior_idx);
void gcxgcy_to_cxcy(float *cxcy, int prior_idx, float *priors_cxcy);
void cxcy_to_xy(float *xy, float *cxcy);
void localize_objects(void);