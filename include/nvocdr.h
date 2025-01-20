#ifndef __NVOCDR_LIB__
#define __NVOCDR_LIB__
#include <stdint.h>
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
  NCHW,
  NHWC
};

enum OCRNetDecode
{
  CTC,
  Attention,
  Transformer
};

typedef struct
{
  DataFormat input_data_format;
  // OCDNetEngine param
  char* ocdnet_trt_engine_path;
  float ocdnet_binarize_threshold;
  float ocdnet_polygon_threshold;
  float ocdnet_unclip_ratio;
  int ocdnet_max_candidate;
  int32_t ocdnet_infer_input_shape[3];
  // Rectifier param
  bool upsidedown = false;
  //  The text box aspect-ratio (width/height) threshold:
  //  If text box aspect-ratio is smaller than threshold then we will rotate the box. 
  float rotation_threshold = 0.0; 
  // OCRNet param:
  char* ocrnet_trt_engine_path;
  char* ocrnet_dict_file;
  int32_t ocrnet_infer_input_shape[3];
  OCRNetDecode ocrnet_decode = Transformer;
  // Param for clip4str
  char* ocrnet_text_trt_engine_path;
  bool ocrnet_only_alnum = false;
  bool ocrnet_only_lowercase = false;
  char* ocrnet_vocab_file;
  int ocrnet_vocab_size = 32000;
  // common param

} nvOCDRParam;


typedef struct
{
  int shape[4];
  void* mem_ptr;
  Device device_type;
  // DataPrecision precision; 
  // DataFormat data_format;
} nvOCDRInput;

typedef struct
{
  int32_t poly_cnt;
  int32_t polys[MAX_PLY_CNT];
  int32_t ch_len;
  char ch[MAX_CHARACTER_LEN];
  float conf;
} nvOCDROutputBlob;

typedef struct
{
  int32_t batch_size;
  int32_t text_cnt[MAX_BATCH_SIZE];
  nvOCDROutputBlob* text_ptr;
} nvOCDROutputMeta;


nvOCDRp nvOCDR_init(nvOCDRParam param);
nvOCDRStat nvOCDR_inference(nvOCDRInput input, nvOCDROutputMeta* output, nvOCDRp nvocdr_ptr);
nvOCDRStat nvOCDR_high_resolution_inference(nvOCDRInput input, nvOCDROutputMeta* output, nvOCDRp nvocdr_ptr,
                                            float overlap_rate);
void nvOCDR_deinit(nvOCDRp nvocdr_ptr);

#ifdef __cplusplus
}
#endif

#endif
