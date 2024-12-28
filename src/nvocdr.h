#ifndef __NVOCDR_LIB__
#define __NVOCDR_LIB__
#include <stddef.h>
#include <stdint.h>
#define MAX_CHARACTER_LEN 64
#define MAX_PLY_CNT 8
#define MAX_FILE_PATH 200

#ifdef __cplusplus
extern "C" {
#endif

enum DATAFORMAT_TYPE { DATAFORMAT_TYPE_CHW, DATAFORMAT_TYPE_HWC };

enum STRATEGY_TYPE {
  // smart to decide strategy which to use.
  STRATEGY_TYPE_SMART,
  // keep ratio resize short side to input size, then do tile
  STRATEGY_TYPE_RESIZE_TILE,
  // no resize, just do tile
  STRATEGY_TYPE_NORESIZE_TILE,
  // STRATEGY_TYPE_RESIZE
};

typedef struct {
  STRATEGY_TYPE strategy;
  size_t max_candidate;
  float polygon_threshold;
  float binarize_lower_threshold;
  float binarize_upper_threshold;
  float min_pixel_area;
  bool debug_log;
  bool debug_image;
  bool
      all_direction;  // if true, 4 direction will be detected, otherwise only one, will consume more time
} ProcessParam;

typedef struct {
  enum OCD_MODEL_TYPE {
    OCD_MODEL_TYPE_NORMAL,
    OCD_MODEL_TYPE_MIXNET,
  };
  char model_file[MAX_FILE_PATH];
  size_t batch_size = 0;  // bs use to do infer, tune it for device with different mem
  OCD_MODEL_TYPE type;
} nvOCDParam;

typedef struct {
  enum OCR_MODEL_TYPE { OCR_MODEL_TYPE_CTC, OCR_MODEL_TYPE_ATTN, OCR_MODEL_TYPE_CLIP };
  char model_file[MAX_FILE_PATH];
  char dict[MAX_FILE_PATH];
  size_t batch_size =
      0;  // bs use to do infer, tune it for device with different mem, -1 to use engine allowed max
  OCR_MODEL_TYPE type = OCR_MODEL_TYPE_CTC;
} nvOCRParam;

typedef struct {
  // OCDNetEngine param
  nvOCDParam ocd_param;
  // OCRNet param:
  nvOCRParam ocr_param;
  // common param
  ProcessParam process_param;
} nvOCDRParam;

typedef struct {
  size_t height;
  size_t width;
  size_t num_channel;
  // input data is owned by user, user responsible for allocate and deallocated
  void*
      data;  // pointer to data in shape [batch_size, height, width, num_channel] or [batch_size, num_channel, height, width]
  DATAFORMAT_TYPE data_format;  // indicate data format in mem
} nvOCDRInput;

typedef struct {
  float polygon[8];  // x,y, ...
  size_t text_length = 0;
  char text[MAX_CHARACTER_LEN];
  float conf = 0.F;
} Text;

typedef struct {
  size_t num_texts;
  // output data is owned by nvOCDR lib, user "must not" allocate or deallocate
  Text* texts;
} nvOCDROutput;

void* nvOCDR_initialize(const nvOCDRParam& param);
bool nvOCDR_process(void* const nvocdr_handler, const nvOCDRInput& input,
                    nvOCDROutput* const output);
void nvOCDR_release(void* const nvocdr_handler);

void nvOCDR_print_stat(void* const nvocdr_handler);

#ifdef __cplusplus
}
#endif

#endif
