#pragma once

#include <map>
#include <string>
#include <opencv2/opencv.hpp>
#include "nvocdr.h"
#include "TRTEngine.h"

namespace nvocdr
{
static constexpr size_t QUAD = 4;
using QUADANGLE = std::array<cv::Point2f, QUAD>;
    

template <typename Param>
class BaseProcessor {
public:
  BaseProcessor(const Param& param): mParam(param) {};
  virtual bool init() = 0;
  virtual bool infer(bool sync_input, const cudaStream_t& stream);

  virtual cv::Size getInputHW() = 0;
  virtual size_t getBatchSize() = 0;
  
  // no matter what we do inside, just give one buffer for outside to fill
  virtual std::string getInputBufName() = 0;

protected:
  std::map<std::string, std::unique_ptr<TRTEngine>> mEngines;
  Param mParam;
  BufferManager& mBufManager = BufferManager::Instance();
};
    
} // namespace nvocdr

