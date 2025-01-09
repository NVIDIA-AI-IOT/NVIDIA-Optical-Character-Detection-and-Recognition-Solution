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
  STRATEGY_TYPE_RESIZE_FULL,
};

typedef struct {
  STRATEGY_TYPE strategy = STRATEGY_TYPE_SMART;
  size_t max_candidate = 100;
  float polygon_threshold = 0.3F;
  float binarize_lower_threshold = 0.F;
  float binarize_upper_threshold = 0.1F;
  float min_pixel_area = 100;
  bool debug_log = false;
  bool debug_image = false;
  // if true, 4 direction will be detected, otherwise only one, will consume more time
  bool all_direction = false;  
} ProcessParam;

typedef struct {
  enum OCD_MODEL_TYPE {
    OCD_MODEL_TYPE_NORMAL,
    OCD_MODEL_TYPE_MIXNET,
  };
  char model_file[MAX_FILE_PATH] = "";
  size_t batch_size = 0;  // bs use to do infer, tune it for device with different mem
  OCD_MODEL_TYPE type = OCD_MODEL_TYPE_NORMAL;
} nvOCDParam;

typedef struct {
  enum OCR_MODEL_TYPE { OCR_MODEL_TYPE_CTC, OCR_MODEL_TYPE_ATTN, OCR_MODEL_TYPE_CLIP };
  char model_file[MAX_FILE_PATH];
  char vocab_file[MAX_FILE_PATH];
  char dict[MAX_FILE_PATH] = "default";
  // bs use to do infer, tune it for device with different mem, -1 to use engine allowed max
  size_t batch_size =0;  
  OCR_MODEL_TYPE type = OCR_MODEL_TYPE_CTC;
} nvOCRParam;

typedef struct {
  //   size_t height;
  // size_t width;
  // size_t num_channel;
  size_t input_shape[3] = {0, 0, 0}; // c, h, w
  // OCDNetEngine param
  nvOCDParam ocd_param;
  // OCRNet param:
  nvOCRParam ocr_param;
  // common param
  ProcessParam process_param;
} nvOCDRParam;

typedef struct {
  // input data is owned by user, user responsible for allocate and deallocated
  // pointer to data in shape [height, width, num_channel] or [num_channel, height, width]
  // data order on channel should be B G R
  // the buffer size equal to the input_shape set in nvOCDRParam
  void* data;  
  DATAFORMAT_TYPE data_format = DATAFORMAT_TYPE_HWC;  // indicate data format in mem
} nvOCDRInput;

typedef struct {
  float polygon[8];  // x,y, ...
  size_t text_length = 0;
  char text[MAX_CHARACTER_LEN] = "";
  float conf = 0.F;
} Text;

typedef struct {
  size_t num_texts;
  // output data is owned by nvOCDR lib, user "must not" allocate or deallocate
  Text* texts;
} nvOCDROutput;

nvOCDRParam nvOCDR_get_default();
void* nvOCDR_initialize(const nvOCDRParam& param);
bool nvOCDR_process(void* const nvocdr_handler, const nvOCDRInput& input,
                    nvOCDROutput* const output);
void nvOCDR_release(void* const nvocdr_handler);

void nvOCDR_print_stat(void* const nvocdr_handler);

#ifdef __cplusplus
}
#endif

#endif
