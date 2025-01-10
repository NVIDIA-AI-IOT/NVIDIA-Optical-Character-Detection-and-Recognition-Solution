#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "MemManager.h"
#include "OCRProcessor.h"
#include "OCDProcessor.h"
#include "nvocdr.h"
#include "timer.hpp"
#include "kernel.h"

namespace nvocdr {
static constexpr float IMG_MEAN_GRAY = 127.5;
static constexpr float IMG_SCALE_GRAY = 0.00784313;

static constexpr float IMG_MEAN_B = 0.4078705409019608;
static constexpr float IMG_MEAN_G = 0.4575245789019608;
static constexpr float IMG_MEAN_R = 0.4810937817254902;

static constexpr float IMG_MEAN_R_MIXNET = 0.485;
static constexpr float IMG_MEAN_G_MIXNET = 0.456;
static constexpr float IMG_MEAN_B_MIXNET = 0.406;

static constexpr float IMG_MEAN_R_STD_MIXNET = 0.229;
static constexpr float IMG_MEAN_G_STD_MIXNET = 0.224;
static constexpr float IMG_MEAN_B_STD_MIXNET = 0.225;

static constexpr float IMG_MEAN_R_CLIP = 0.4814546;
static constexpr float IMG_MEAN_G_CLIP = 0.4578275;
static constexpr float IMG_MEAN_B_CLIP = 0.4082107;

static constexpr float IMG_MEAN_R_STD_CLIP = 0.26862954;
static constexpr float IMG_MEAN_G_STD_CLIP = 0.26130258;
static constexpr float IMG_MEAN_B_STD_CLIP = 0.27577711;

static constexpr size_t NUM_WARMUP_RUNS = 10;
static constexpr size_t TIME_HISTORY_SIZE = 100;

static constexpr size_t C_IDX = 0;
static constexpr size_t H_IDX = 1;
static constexpr size_t W_IDX = 2;

static constexpr COLOR_PREPROC_PARAM OCD_NORMAL_PREPROR_PARAM {
    .rgb_scale = 255.F,
    .r_mean = IMG_MEAN_B,
    .g_mean = IMG_MEAN_G,
    .b_mean = IMG_MEAN_R,
    .r_std = 1,
    .g_std = 1,
    .b_std = 1
};

static constexpr COLOR_PREPROC_PARAM OCD_MIXNET_PREPROR_PARAM {
    .rgb_scale = 255.F,
    .r_mean = IMG_MEAN_R_MIXNET,
    .g_mean = IMG_MEAN_G_MIXNET,
    .b_mean = IMG_MEAN_B_MIXNET,
    .r_std = IMG_MEAN_R_STD_MIXNET,
    .g_std = IMG_MEAN_G_STD_MIXNET,
    .b_std = IMG_MEAN_B_STD_MIXNET
};

// reuse COLOR_PREPROC_PARAM
static constexpr COLOR_PREPROC_PARAM OCR_PREPROC_GRAY_PARAM {
  .rgb_scale = IMG_SCALE_GRAY,
  .r_mean = IMG_MEAN_GRAY
};

static constexpr COLOR_PREPROC_PARAM OCR_PREPROC_CLIP_PARAM {
    .rgb_scale = 255.F,
    .r_mean = IMG_MEAN_R_CLIP,
    .g_mean = IMG_MEAN_G_CLIP,
    .b_mean = IMG_MEAN_B_CLIP,
    .r_std = IMG_MEAN_R_STD_CLIP,
    .g_std = IMG_MEAN_G_STD_CLIP,
    .b_std = IMG_MEAN_B_STD_CLIP
};

enum INPUT_NORM_STYLE {
  INPUT_NORM_STYLE_MIXNET, // rgb + its only mean + std
  INPUT_NORM_STYLE_DCN, // bgr + its only mean + std=1
  INPUT_NORM_STYLE_CLIP, // rgb + its only mean + std
  INPUT_NORM_STYLE_GRAY, // gray + normalized
};

static constexpr char ORIGIN_INPUT_BUF[] = "input_image";

class nvOCDR {
 public:
  nvOCDR(const nvOCDRParam& param);
  ~nvOCDR() {
    mBufManager.releaseAllBuffers();
  }
  void process(const nvOCDRInput& input, nvOCDROutput* const output);
  void printTimeStat();

 private:
  void initBuffers();
  void processTile(const nvOCDRInput& input);

  void handleStrategy();

  void getTilePlan(size_t input_w, size_t input_h, size_t raw_w, size_t raw_h, size_t stride);
  void preprocessOCDTile(size_t start, size_t end);
  void preprocessOCDTileGPU(size_t start, size_t end);
  void postprocessOCDTile(size_t start, size_t end);

  void selectOCRCandidates();
  void preprocessOCR(size_t start, size_t end, size_t bl_pt_idx);
  void postprocessOCR(size_t start, size_t end);

  void restoreImage(const nvOCDRInput& input);
  void setOutput(nvOCDROutput* const output);

  std::unique_ptr<OCRProcessor> mOCRProcessor;
  std::unique_ptr<OCDProcessor> mOCDProcessor;

  BufferManager& mBufManager = BufferManager::Instance();

  cv::Mat mInputImage;        

  cv::Mat mOCDScoreMap;           
  cv::Mat mOCDOutputMask;         
  cv::Mat mOCDValidCntMap;   

  std::vector<size_t> mOCRDirections;

  std::vector<cv::Rect> mTiles;

  /// @brief text output
  std::vector<Text> mTexts;
  /// @brief output text number 
  size_t mNumTexts;
  /// @brief output for 
  std::vector<QUADANGLE> mQuadPts;


  /// @brief ocd engine expected size
  cv::Size mOCDInputSize;
  /// @brief ocr engine expected size
  cv::Size mOCRInputSize;

  /// @brief input channe number 
  size_t mInputChannels = 3;
  /// @brief origin size for input image
  cv::Size mRawInputSize;
  /// @brief destionate size for ocd
  cv::Size mDestinateSize;

  // todo(shuohanc) use difference stream to improve performance more
  cudaStream_t mStream;
  nvOCDRParam mParam;

  // Timer<TIME_HISTORY_SIZE> mPreprocessTimer;
  Timer<TIME_HISTORY_SIZE> mOCDTimer;
  Timer<TIME_HISTORY_SIZE> mSelectProcessTimer;
  Timer<TIME_HISTORY_SIZE> mOCRTimer;
  Timer<TIME_HISTORY_SIZE> mE2ETimer;
};
}  // namespace nvocdr
