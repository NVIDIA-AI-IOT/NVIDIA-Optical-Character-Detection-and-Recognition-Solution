#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "MemManager.h"
#include "OCDNetEngine.h"
#include "OCRNetEngine.h"
#include "nvocdr.h"
#include "opencv2/opencv.hpp"
#include "timer.hpp"

namespace nvocdr {
static constexpr float IMG_MEAN_GRAY = 127.5;
static constexpr float IMG_SCALE_GRAY = 0.00784313;
static constexpr float IMG_MEAN_B = 104.00698793;
static constexpr float IMG_MEAN_G = 116.66876762;
static constexpr float IMG_MEAN_R = 122.67891434;
static constexpr float IMG_SCALE_BRG = 0.00392156;

static constexpr size_t NUM_WARMUP_RUNS = 10;
static constexpr size_t TIME_HISTORY_SIZE = 100;

class nvOCDR {
 public:
  nvOCDR(const nvOCDRParam& param);
  void process(const nvOCDRInput& input, nvOCDROutput* const output);
  void printTimeStat();

 private:
  void processTile(const nvOCDRInput& input);

  void handleStrategy(const nvOCDRInput& input);

  void getTilePlan(size_t input_w, size_t input_h, size_t raw_w, size_t raw_h, size_t stride);
  void preprocessOCDTile(size_t start, size_t end);
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

  /** origin size, resized size */
  std::pair<cv::Size, cv::Size> mResizeInfo;
  std::vector<cv::Rect> mTiles;

  std::vector<QUADANGLE> mQuadPts;

  std::vector<Text> mTexts;
  size_t mNumTexts;

  cudaStream_t mStream;
  nvOCDRParam mParam;
  Timer<TIME_HISTORY_SIZE> mOCDTimer;
  Timer<TIME_HISTORY_SIZE> mSelectProcessTimer;
  Timer<TIME_HISTORY_SIZE> mOCRTimer;
  Timer<TIME_HISTORY_SIZE> mE2ETimer;
};
}  // namespace nvocdr
