#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "MemManager.h"
#include "NvInfer.h"
#include "nvocdr.h"

namespace nvocdr {

using namespace nvinfer1;

size_t volume(const Dims& dim);

struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    delete obj;
  }
};

inline std::ostream& operator<<(std::ostream& o, const Dims& dims) {
  o << '[';
  for (size_t i = 0; i < dims.nbDims; ++i) {
    o << (i > 0 ? "," : "") << dims.d[i];
  }
  o << ']';
  return o;
}

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, InferDeleter>;

template <typename Param>
class TRTEngine {
 public:
  
  TRTEngine(const char name[], const Param& param) : mName(name), mParam(param) {};

  bool init();
  bool initEngine();
  virtual bool customInit() = 0;
  bool postInit();

  // Do TRT-based inference with processed input and get output
  bool infer(const cudaStream_t& stream = 0);
  bool syncMemory(bool input, bool host2device, const cudaStream_t& stream);

  inline std::string getBufName(const std::string& name) { return mName + "_" + name; }
  inline size_t getInputH() { return mInputH; };
  inline size_t getInputW() { return mInputW; };
  inline size_t getBatchSize() { return mBatchSize; }
  nvinfer1::Dims getOutputDims(const std::string& name);

 protected:
  void setupInput(const std::string& input_name, const Dims& dims, bool host_buf = false);
  void setupOutput(const std::string& output_name, const Dims& dims, bool host_buf = false);
  ~TRTEngine() = default;


  Param mParam;
  std::string mName;
  BufferManager& mBufManager = BufferManager::Instance();

 private:
  TRTUniquePtr<IRuntime> mRuntime;
  TRTUniquePtr<ICudaEngine> mEngine;
  TRTUniquePtr<IExecutionContext> mContext;

  size_t mBatchSize = 1U;
  size_t mInputH;
  size_t mInputW;

  std::vector<std::string> mInputNames;
  std::vector<std::string> mOutputNames;
  std::map<std::string, nvinfer1::Dims> mInputDims;
  std::map<std::string, nvinfer1::Dims> mOutputDims;
};

using OCRTRTEngine = TRTEngine<nvOCRParam>;
using OCDTRTEngine = TRTEngine<nvOCDParam>;

}  // namespace nvocdr