/******************************************************************************
 * post_process.c — TinierSSD (MAX78000 facedet)
 *
 * Differences from stock cam02 demo:
 *   - softmax(): removed the early-exit guard `if (calc_softmax==0) continue`,
 *     which silently zeroed every prior because background raw score always
 *     dominates face raw score in this 3-class model.
 *   - get_priors(): emits a "SCORE:..." diagnostic line per frame for the
 *     PC-side script to parse and overlay.
 *   - nms(): removed the static max_score_seen debug print (caused stream
 *     noise that desynced the PC parser).
 ******************************************************************************/

#include "post_process.h"
#include "mxc_device.h"
#include "mxc.h"

#define S_MODULE_NAME "post_proc"

uint8_t box[4]; /* x1, y1, x2, y2 */

const int dims[NUM_SCALES][2] = { { 28, 21 }, { 7, 5 } };
const float scales[NUM_SCALES] = { 0.35715f, 0.7143f };
const float ars[NUM_ARS] = { 0.9f, 0.75f };

static int8_t   prior_locs[LOC_DIM * NUM_PRIORS];
static int8_t   prior_cls[NUM_CLASSES * NUM_PRIORS];
static uint16_t prior_cls_softmax[NUM_CLASSES * NUM_PRIORS] = { 0 };

static uint16_t nms_scores[NUM_CLASSES - 2][MAX_PRIORS];
static uint16_t nms_indices[NUM_CLASSES - 2][MAX_PRIORS];
static uint8_t  nms_removed[NUM_CLASSES - 2][MAX_PRIORS] = { 0 };
static int      num_nms_priors[NUM_CLASSES - 2] = { 0 };
volatile uint8_t face_detected = 0;

int get_prior_idx(int ar_idx, int scale_idx, int rel_idx)
{
    int prior_idx = 0;
    for (int s = 0; s < scale_idx; ++s)
        prior_idx += NUM_ARS * MULT(dims[s][0], dims[s][1]);
    prior_idx += NUM_ARS * rel_idx + ar_idx;
    return prior_idx;
}

void get_indices(int *ar_idx, int *scale_idx, int *rel_idx, int prior_idx)
{
    int s, prior_count = 0;
    for (s = 0; s < NUM_SCALES; ++s) {
        prior_count += (NUM_ARS * MULT(dims[s][0], dims[s][1]));
        if (prior_idx < prior_count) { *scale_idx = s; break; }
    }
    int in_scale_idx = prior_idx;
    for (s = 0; s < *scale_idx; ++s)
        in_scale_idx -= (NUM_ARS * MULT(dims[s][0], dims[s][1]));
    *ar_idx  = in_scale_idx % NUM_ARS;
    *rel_idx = in_scale_idx / NUM_ARS;
}

/* Unconditional softmax over all priors. */
void softmax(void)
{
    int i, ch;
    float sum;

    memset(prior_cls_softmax, 0, sizeof(prior_cls_softmax));
    for (i = 0; i < NUM_PRIORS; ++i) {
        sum = 0.f;
        for (ch = 0; ch < NUM_CLASSES; ++ch)
            sum += (float)exp(prior_cls[i * NUM_CLASSES + ch] / 128.0);
        for (ch = 0; ch < NUM_CLASSES; ++ch)
            prior_cls_softmax[i * NUM_CLASSES + ch] =
                (uint16_t)(65536.0 * (float)exp(prior_cls[i * NUM_CLASSES + ch] / 128.0) / sum);
    }
}

void get_prior_locs(void)
{
    int8_t *loc_addr = (int8_t *)0x50400000;
    int ar_idx, scale_idx, rel_idx, prior_idx, prior_count;

    for (ar_idx = 0; ar_idx < NUM_ARS; ++ar_idx) {
        int8_t *loc_addr_temp = loc_addr;
        for (scale_idx = 0; scale_idx < NUM_SCALES; ++scale_idx) {
            prior_count = MULT(dims[scale_idx][0], dims[scale_idx][1]);
            for (rel_idx = 0; rel_idx < prior_count; ++rel_idx) {
                prior_idx = get_prior_idx(ar_idx, scale_idx, rel_idx);
                memcpy(&prior_locs[LOC_DIM * prior_idx], loc_addr_temp, LOC_DIM);
                loc_addr_temp += LOC_DIM;
            }
        }
        loc_addr += 0x8000;
    }
}

void get_prior_cls(void)
{
    int8_t *cl_addr = (int8_t *)0x50410000;
    int ar_idx, cl_idx, scale_idx, rel_idx, prior_idx, prior_count;

    for (scale_idx = 0; scale_idx < NUM_SCALES; ++scale_idx) {
        prior_count = MULT(dims[scale_idx][0], dims[scale_idx][1]);
        for (ar_idx = 0; ar_idx < NUM_ARS; ++ar_idx) {
            int8_t *cl_addr_temp = cl_addr + ar_idx * 2;
            for (rel_idx = 0; rel_idx < prior_count; ++rel_idx) {
                for (cl_idx = 0; cl_idx < NUM_CLASSES - 1; cl_idx += 1) {
                    cl_addr_temp += cl_idx;
                    prior_idx = get_prior_idx(ar_idx, scale_idx, rel_idx);
                    memcpy(&prior_cls[NUM_CLASSES * prior_idx + cl_idx], cl_addr_temp, 1);
                }
                cl_addr_temp += 3;
            }
        }
        cl_addr = (int8_t *)0x50410930;
    }
    softmax();
}

void get_priors(void)
{
    get_prior_locs();
    get_prior_cls();

    /* Per-frame diagnostic. Python parses "SCORE:" lines.
     * Format: SCORE:max_cls0:max_cls1:max_sm1:threshold */
    int8_t   max_cls0 = -128, max_cls1 = -128;
    uint16_t max_sm1  = 0;
    for (int i = 0; i < NUM_PRIORS; i++) {
        if (prior_cls[i * NUM_CLASSES + 0] > max_cls0)
            max_cls0 = prior_cls[i * NUM_CLASSES + 0];
        if (prior_cls[i * NUM_CLASSES + 1] > max_cls1)
            max_cls1 = prior_cls[i * NUM_CLASSES + 1];
        if (prior_cls_softmax[i * NUM_CLASSES + 1] > max_sm1)
            max_sm1 = prior_cls_softmax[i * NUM_CLASSES + 1];
    }
    printf("SCORE:%d:%d:%u:%u\n",
           (int)max_cls0, (int)max_cls1,
           (unsigned)max_sm1, (unsigned)MIN_CLASS_SCORE);
}

float calculate_IOU(float *box1, float *box2)
{
    float x_left  = MAX(box1[0], box2[0]);
    float y_top   = MAX(box1[1], box2[1]);
    float x_right = MIN(box1[2], box2[2]);
    float y_bot   = MIN(box1[3], box2[3]);
    if (x_right < x_left || y_bot < y_top) return 0.0f;
    float inter = (x_right - x_left) * (y_bot - y_top);
    float a1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float a2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    return inter / (a1 + a2 - inter);
}

void get_cxcy(float *cxcy, int prior_idx)
{
    int i, scale_idx, ar_idx, rel_idx, cx, cy;
    get_indices(&ar_idx, &scale_idx, &rel_idx, prior_idx);
    cy = rel_idx / dims[scale_idx][1];
    cx = rel_idx % dims[scale_idx][1];
    cxcy[0] = (float)((float)(cx + 0.5) / dims[scale_idx][1]);
    cxcy[1] = (float)((float)(cy + 0.5) / dims[scale_idx][0]);
    cxcy[2] = scales[scale_idx] * (float)sqrt(ars[ar_idx]);
    cxcy[3] = scales[scale_idx] / (float)sqrt(ars[ar_idx]);
    for (i = 0; i < 4; ++i) {
        cxcy[i] = MAX(0.0f, cxcy[i]);
        cxcy[i] = MIN(cxcy[i], 1.0f);
    }
}

void gcxgcy_to_cxcy(float *cxcy, int prior_idx, float *priors_cxcy)
{
    float gcxgcy[4];
    for (int i = 0; i < 4; i++)
        gcxgcy[i] = (float)prior_locs[4 * prior_idx + i] / 128.0f;
    cxcy[0] = priors_cxcy[0] + gcxgcy[0] * priors_cxcy[2] / 10.0f;
    cxcy[1] = priors_cxcy[1] + gcxgcy[1] * priors_cxcy[3] / 10.0f;
    cxcy[2] = (float)exp(gcxgcy[2] / 5.0f) * priors_cxcy[2];
    cxcy[3] = (float)exp(gcxgcy[3] / 5.0f) * priors_cxcy[3];
}

void cxcy_to_xy(float *xy, float *cxcy)
{
    xy[0] = cxcy[0] - cxcy[2] / 2;
    xy[1] = cxcy[1] - cxcy[3] / 2;
    xy[2] = cxcy[0] + cxcy[2] / 2;
    xy[3] = cxcy[1] + cxcy[3] / 2;
}

void insert_val(uint16_t val, uint16_t *arr, int arr_len, int idx)
{
    if (arr_len < MAX_PRIORS) arr[arr_len] = arr[arr_len - 1];
    for (int j = (arr_len - 1); j > idx; --j) arr[j] = arr[j - 1];
    arr[idx] = val;
}

void insert_idx(uint16_t val, uint16_t *arr, int arr_len, int idx)
{
    if (arr_len < MAX_PRIORS) arr[arr_len] = arr[arr_len - 1];
    for (int j = (arr_len - 1); j > idx; --j) arr[j] = arr[j - 1];
    arr[idx] = val;
}

void insert_nms_prior(uint16_t val, int idx, uint16_t *val_arr, uint16_t *idx_arr, int *arr_len)
{
    if ((*arr_len == 0) ||
        ((val <= val_arr[*arr_len - 1]) && (*arr_len != MAX_PRIORS))) {
        val_arr[*arr_len] = val;
        idx_arr[*arr_len] = idx;
    } else {
        for (int i = 0; i < *arr_len; ++i) {
            if (val > val_arr[i]) {
                insert_val(val, val_arr, *arr_len, i);
                insert_idx(idx, idx_arr, *arr_len, i);
                break;
            }
        }
    }
    *arr_len = MIN((*arr_len + 1), MAX_PRIORS);
}

void reset_nms(void)
{
    for (int cl = 0; cl < NUM_CLASSES - 2; ++cl) {
        num_nms_priors[cl] = 0;
        for (int p = 0; p < MAX_PRIORS; ++p) {
            nms_scores[cl][p]  = 0;
            nms_indices[cl][p] = 0;
            nms_removed[cl][p] = 0;
        }
    }
}

void nms(void)
{
    int prior_idx, class_idx, nms_idx1, nms_idx2, prior1_idx, prior2_idx;
    uint16_t cls_prob;
    float prior_cxcy1[4], prior_cxcy2[4];
    float cxcy1[4], cxcy2[4];
    float xy1[4],   xy2[4];

    reset_nms();

    for (prior_idx = 0; prior_idx < NUM_PRIORS; ++prior_idx) {
        for (class_idx = 0; class_idx < (NUM_CLASSES - 2); ++class_idx) {
            cls_prob = prior_cls_softmax[prior_idx * NUM_CLASSES + class_idx + 1];
            if (cls_prob < MIN_CLASS_SCORE) continue;
            insert_nms_prior(cls_prob, prior_idx,
                             nms_scores[class_idx], nms_indices[class_idx],
                             &num_nms_priors[class_idx]);
        }
    }

    for (class_idx = 0; class_idx < (NUM_CLASSES - 2); ++class_idx) {
        for (nms_idx1 = 0; nms_idx1 < num_nms_priors[class_idx]; ++nms_idx1) {
            if (nms_removed[class_idx][nms_idx1] != 1 &&
                nms_idx1 != num_nms_priors[class_idx] - 1) {
                for (nms_idx2 = nms_idx1 + 1; nms_idx2 < num_nms_priors[class_idx]; ++nms_idx2) {
                    prior1_idx = nms_indices[class_idx][nms_idx1];
                    prior2_idx = nms_indices[class_idx][nms_idx2];
                    get_cxcy(prior_cxcy1, prior1_idx);
                    get_cxcy(prior_cxcy2, prior2_idx);
                    gcxgcy_to_cxcy(cxcy1, prior1_idx, prior_cxcy1);
                    gcxgcy_to_cxcy(cxcy2, prior2_idx, prior_cxcy2);
                    cxcy_to_xy(xy1, cxcy1);
                    cxcy_to_xy(xy2, cxcy2);
                    if (calculate_IOU(xy1, xy2) > MAX_ALLOWED_OVERLAP)
                        nms_removed[class_idx][nms_idx2] = 1;
                }
            }
        }
    }
}

void box_sanity_check(float *xy)
{
    for (int i = 0; i < 4; ++i) {
        if (xy[i] < 0.0f) xy[i] = 0.0f;
        else if (xy[i] > 1.0f) xy[i] = 1.0f;
    }
}

void localize_objects(void)
{
    float prior_cxcy[4], cxcy[4], xy[4];
    int class_idx, prior_idx, global_prior_idx;

    nms();

    for (class_idx = 0; class_idx < (NUM_CLASSES - 2); ++class_idx) {
        for (prior_idx = 0; prior_idx < num_nms_priors[class_idx]; ++prior_idx) {
            if (nms_removed[class_idx][prior_idx] == 1) continue;

            global_prior_idx = nms_indices[class_idx][prior_idx];
            get_cxcy(prior_cxcy, global_prior_idx);
            gcxgcy_to_cxcy(cxcy, global_prior_idx, prior_cxcy);
            cxcy_to_xy(xy, cxcy);
            box_sanity_check(xy);

            box[0] = (uint8_t)(IMAGE_SIZE_X * xy[0]);
            box[1] = (uint8_t)(IMAGE_SIZE_Y * xy[1]);
            box[2] = (uint8_t)(IMAGE_SIZE_X * xy[2]);
            box[3] = (uint8_t)(IMAGE_SIZE_Y * xy[3]);

            face_detected = 1;
        }
    }
}