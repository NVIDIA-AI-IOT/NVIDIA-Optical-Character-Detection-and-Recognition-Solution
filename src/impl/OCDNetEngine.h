#pragma once

#include <string>

#include <opencv2/opencv.hpp>

#include "MemManager.h"
#include "TRTEngine.h"
#include "nvocdr.h"

// #define OCD_DEBUG

namespace nvocdr {
static char OCD_PREFIX[] = "OCD";
static char OCDNET_INPUT[] = "input";
static constexpr size_t QUAD = 4;

using QUADANGLE = std::array<cv::Point2f, QUAD>;

// constexpr char OCDNET_OUTPUT[] = "pred";

class OCDNetEngine : public OCDTRTEngine {
 public:
  bool customInit() final;
  OCDNetEngine(const char name[], const nvOCDParam& param) : OCDTRTEngine(name, param) {};
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
  // todo (shuohanc) put in attri temporarily, to save mem alloc/dealloc
  std::vector<std::vector<cv::Point>> mTextCntrCandidates;
};
}  // namespace nvocdr
