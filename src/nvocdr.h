#ifndef __NVOCDR_LIB__
#define __NVOCDR_LIB__
#include <stdint.h>
#include <stddef.h>
#define MAX_CHARACTER_LEN 64
#define MAX_PLY_CNT 8
// The nvOCDR lib's batch size at images level.
// Maximum 64 images can be feed to nvOCDR at once.
#define MAX_BATCH_SIZE 64

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
} ProcessParam;


typedef struct {
  char *engine_file;
  char *onnx_file;
  size_t batch_size = 0; // bs use to do infer, tune it for device with different mem
  float binarize_threshold;
  float polygon_threshold;
  float unclip_ratio;
  size_t max_candidate;
} nvOCDParam;


typedef struct {
  char *engine_file;
  char *onnx_file;
  char *dict_file;
  size_t batch_size = 0; // bs use to do infer, tune it for device with different mem, -1 to use engine allowed max
  OCRNetDecode mode = CTC;
} nvOCRParam;

// typedef struct {


// } nvRectifyParam;

typedef struct
{
  DataFormat input_data_format;
  // OCDNetEngine param
  nvOCDParam ocd_param;
  // Rectifier param
  bool upsidedown = false;
  //  The text box aspect-ratio (width/height) threshold:
  //  If text box aspect-ratio is smaller than threshold then we will rotate the box. 
  float rotation_threshold = 0.0; 
  // OCRNet param:
  nvOCRParam ocr_param;
  // common param
  ProcessParam process_param;
} nvOCDRParam;


// typedef struct
// {
//   DataFormat input_data_format;
//   // OCDNetEngine param
//   char* ocdnet_trt_engine_path;
//   float ocdnet_binarize_threshold;
//   float ocdnet_polygon_threshold;
//   float ocdnet_unclip_ratio;
//   int ocdnet_max_candidate;
//   int32_t ocdnet_infer_input_shape[3];
//   // Rectifier param
//   bool upsidedown = false;
//   //  The text box aspect-ratio (width/height) threshold:
//   //  If text box aspect-ratio is smaller than threshold then we will rotate the box. 
//   float rotation_threshold = 0.0; 
//   // OCRNet param:
//   char* ocrnet_trt_engine_path;
//   char* ocrnet_dict_file;
//   int32_t ocrnet_infer_input_shape[3];
//   OCRNetDecode ocrnet_decode = CTC;
//   // common param

// } nvOCDRParam;


typedef struct
{
  size_t height;
  size_t width;
  size_t num_channel;
  // data should be in RGB order
  void* data;  // pointer to data in shape [batch_size, height, width, num_channel] or [batch_size, num_channel, height, width]
  // Device device_type;
  // DataPrecision precision; 
  DataFormat data_format; // indicate data format in mem
} nvOCDRInput;


typedef struct
{
  size_t poly_cnt;
  size_t polys[MAX_PLY_CNT];
  size_t ch_len;
  char ch[MAX_CHARACTER_LEN];
  float conf;
} nvOCDROutput;

// typedef struct
// {
//   size_t num_i
//   size_t num_text_per_image[100]; 
//   nvOCDROutputBlob* blob_data;
// } nvOCDROutputMeta;

// nvOCDRp nvOCDR_init(nvOCDRParam param);
// nvOCDRStat nvOCDR_inference(nvOCDRInput input, nvOCDROutputMeta* output, nvOCDRp nvocdr_ptr);
// nvOCDRStat nvOCDR_high_resolution_inference(nvOCDRInput input, nvOCDROutputMeta* output, nvOCDRp nvocdr_ptr,
//                                             float overlap_rate);
// void nvOCDR_deinit(nvOCDRp nvocdr_ptr);


nvOCDRp nvOCDR_initialize(const nvOCDRParam& param);
// nvOCDRStat nvOCDR_add_input(void * const nvocdr_handler, const nvOCDRInput& input);
// nvOCDRStat nvOCDR_get_output(void * const nvocdr_handler, const nvOCDROutputMeta* output);

nvOCDRStat nvOCDR_process(void * const nvocdr_handler, const nvOCDRInput& input, const nvOCDROutput* output);

void nvOCDR_release(void * const nvocdr_handler);
                                            
#ifdef __cplusplus
}
#endif

#endif
