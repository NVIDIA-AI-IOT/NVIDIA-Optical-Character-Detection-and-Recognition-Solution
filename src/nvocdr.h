#ifndef __NVOCDR_LIB__
#define __NVOCDR_LIB__
#include <stdint.h>
#include <stddef.h>
#define MAX_CHARACTER_LEN 64
#define MAX_PLY_CNT 8
// The nvOCDR lib's batch size at images level.

#ifdef __cplusplus
extern "C"{
#endif

typedef void* nvOCDRp;

enum nvOCDRStat
{
  SUCCESS,
  FAIL
};

enum Device
{
  CPU,
  GPU
};


enum DataFormat
{
  CHW,
  HWC
};

enum OCRNetDecode
{
  CTC,
  Attention,
  CLIP
};

enum StratetyType 
{
  SMART,
  RESIZE
};

typedef struct 
{
  StratetyType strategy;
  size_t max_candidate;
  float polygon_threshold;
  float binarize_threshold;
  float unclip_ratio;
  float min_pixel_area;
} ProcessParam;


typedef struct {
  char *engine_file;
  char *onnx_file;
  size_t batch_size = 0; // bs use to do infer, tune it for device with different mem
} nvOCDParam;


typedef struct {
  char *engine_file;
  char *onnx_file;
  char *dict_file;
  size_t batch_size = 0; // bs use to do infer, tune it for device with different mem, -1 to use engine allowed max
  OCRNetDecode mode = CTC;
} nvOCRParam;

typedef struct
{
  DataFormat input_data_format;
  // OCDNetEngine param
  nvOCDParam ocd_param;
  // OCRNet param:
  nvOCRParam ocr_param;
  // common param
  ProcessParam process_param;
} nvOCDRParam;


typedef struct
{
  size_t height;
  size_t width;
  size_t num_channel;
  // data should be in RGB order
  void* data;  // pointer to data in shape [batch_size, height, width, num_channel] or [batch_size, num_channel, height, width]
  DataFormat data_format; // indicate data format in mem
} nvOCDRInput;



typedef struct {
  float polygon[8]; // x,y, ...
  size_t text_length = 0;
  char text[MAX_CHARACTER_LEN];
  float angle;
  float conf;
} Text;

typedef struct
{
  size_t num_texts;
  Text* texts;
} nvOCDROutput;


nvOCDRp nvOCDR_initialize(const nvOCDRParam& param);
nvOCDRStat nvOCDR_process(void * const nvocdr_handler, const nvOCDRInput& input,  nvOCDROutput* const output);
void nvOCDR_release(void * const nvocdr_handler);
                                            
#ifdef __cplusplus
}
#endif

#endif
