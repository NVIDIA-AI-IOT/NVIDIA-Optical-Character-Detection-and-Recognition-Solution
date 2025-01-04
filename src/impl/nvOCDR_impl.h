#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "MemManager.h"
#include "OCDNetEngine.h"
#include "OCRNetEngine.h"
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
static constexpr float IMG_SCALE_BRG = 0.00392156;

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

static constexpr GRAY_PREPROC_PARAM OCR_PREPROC_PARAM {
  .gray_scale = IMG_SCALE_GRAY,
  .mean = IMG_MEAN_GRAY
};


class nvOCDR {
 public:
  nvOCDR(const nvOCDRParam& param);
  ~nvOCDR() {
    mBufManager.releaseAllBuffers();
  }
  void process(const nvOCDRInput& input, nvOCDROutput* const output);
  void printTimeStat();

  void testfunc(cv::Mat & input_image);

 private:
  void initBuffers();
  void processTile(const nvOCDRInput& input);

  void handleStrategy(const nvOCDRInput& input);

  void getTilePlan(size_t input_w, size_t input_h, size_t raw_w, size_t raw_h, size_t stride);
  void preprocessOCDTile(size_t start, size_t end);
  void preprocessOCDTileGPU(size_t start, size_t end);
  void postprocessOCDTile(size_t start, size_t end);

  void selectOCRCandidates();
  void preprocessOCR(size_t start, size_t end, size_t bl_pt_idx);
  void postprocessOCR(size_t start, size_t end);

  void preprocessInputImage();
  void restoreImage(const nvOCDRInput& input);
  void setOutput(nvOCDROutput* const output);
  cv::Mat denormalizeGray(const cv::Mat& input);

  std::unique_ptr<OCRNetEngine> mOCRNet;
  std::unique_ptr<OCDNetEngine> mOCDNet;
  BufferManager& mBufManager = BufferManager::Instance();

  cv::Mat mInputImage;         // origin input image, aribitrial size
  cv::Mat mInputImageResized;  // origin input image, aribitrial size

  cv::Mat mInputImageResized32F;  // input image, float32, resized
  cv::Mat mOCDScoreMap;           // OCD score map, size = resized
  cv::Mat mOCDOutputMask;         // OCD output mask, size = resized
  cv::Mat mOCDValidCntMap;        // OCD valid cnt map, size = resized

  cv::Mat mInputGrayImage;  // input image, float32 and gray, size = mInputImage

  std::vector<size_t> mOCRDirections;

  /** origin size, resized size */
  std::pair<cv::Size, cv::Size> mResizeInfo;
  std::vector<cv::Rect> mTiles;

  std::vector<QUADANGLE> mQuadPts;

  std::vector<Text> mTexts;
  size_t mNumTexts;

  std::array<size_t, 3> mInputShape; // c, h, w

  cudaStream_t mStream;
  nvOCDRParam mParam;
  Timer<TIME_HISTORY_SIZE> mPreprocessTimer;
  Timer<TIME_HISTORY_SIZE> mOCDTimer;
  Timer<TIME_HISTORY_SIZE> mSelectProcessTimer;
  Timer<TIME_HISTORY_SIZE> mOCRTimer;
  Timer<TIME_HISTORY_SIZE> mE2ETimer;
  Timer<TIME_HISTORY_SIZE> mTmpTimer;
};
}  // namespace nvocdr
