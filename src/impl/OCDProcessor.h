#pragma once

#include <string>

#include <opencv2/opencv.hpp>

#include "MemManager.h"
#include "TRTEngine.h"
#include "base.h"
#include "nvocdr.h"

namespace nvocdr {
// static char OCD_PREFIX[] = "OCD";
static char OCDNET_INPUT[] = "input";
static char OCD_MODEL[] = "OCD";
// 
static char OCD_MIXNET_OUTPUT[] = "fy_preds";
static char OCD_NORMAL_OUTPUT[] = "pred";

class OCDProcessor : public BaseProcessor<nvOCDParam> {
 public:
  bool init() final;
  OCDProcessor(const nvOCDParam& param);
  cv::Size getInputHW() final;
  std::string getInputBufName() final;
  size_t getBatchSize() final;

  float* getMaskOutputBuf();
  size_t getOutputChannelIdx();
  size_t getOutputChannels();

  void computeTextCandidates(const cv::Mat& mask, std::vector<QUADANGLE>* const quads,
                             std::vector<Text>* const texts, size_t* num_text,
                             const ProcessParam& process_param);

 private:
  void computeTextCandidatesNormal(const cv::Mat& mask, std::vector<QUADANGLE>* const quads,
                                   std::vector<Text>* const texts, size_t* num_text,
                                   const ProcessParam& process_param);
  void computeTextCandidatesMixNet(const cv::Mat& mask, std::vector<QUADANGLE>* const quads,
                                   std::vector<Text>* const texts, size_t* num_text,
                                   const ProcessParam& process_param);
  std::string mOutputName;
  // todo (shuohanc) put in attri temporarily, to save mem alloc/dealloc overhead
  std::vector<std::vector<cv::Point>> mTextCntrCandidates;
};
}  // namespace nvocdr
