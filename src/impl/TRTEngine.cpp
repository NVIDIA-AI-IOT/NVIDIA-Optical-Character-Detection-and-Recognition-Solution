#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <glog/logging.h>

#include "TRTEngine.h"

using namespace nvinfer1;

namespace nvocdr {
class Logger : public ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    // suppress info-level messages
    if (severity <= Severity::kERROR){
      std::cout << msg << '\n';}
  }
} logger;

bool TRTEngine::postInit() {
  for (auto const& name : mOutputNames) {
    if (!mBufManager.checkBufferExist(getBufName(name))) {
      LOG(WARNING) << "buffer for '" << name << "' not allocated, allocate to max";
      setupOutput(name, {}, false);  // remove in model export
    }
  }
  return true;
}


bool TRTEngine::initEngine() {

  initLibNvInferPlugins(reinterpret_cast<void*>(&logger), "");
  mRuntime = std::move(TRTUniquePtr<IRuntime>(createInferRuntime(logger)));
  //load binary engine:
  LOG(INFO) << "load engine from:" << mModelPath;
  std::ifstream engineFile(mModelPath, std::ios::binary);
  if (!engineFile.good()) {
    LOG(ERROR) << "Error reading serialized TensorRT engine: " << mModelPath;
  }
  engineFile.seekg(0, std::ifstream::end);
  const int64_t fsize = engineFile.tellg();
  engineFile.seekg(0, std::ifstream::beg);

  std::vector<char> engineBlob(fsize);
  engineFile.read(engineBlob.data(), fsize);
  if (!engineFile.good()) {
    LOG(INFO) << "Error reading serialized TensorRT engine: " << mModelPath;
    throw std::runtime_error("Error reading serialized TensorRT engine");
  }

  mEngine = std::move(TRTUniquePtr<ICudaEngine>(
      mRuntime->deserializeCudaEngine(engineBlob.data(), engineBlob.size())));
  mContext = std::move(TRTUniquePtr<IExecutionContext>(mEngine->createExecutionContext()));

  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const std::string tensor_name(mEngine->getIOTensorName(i));

    if (mEngine->getTensorIOMode(tensor_name.c_str()) == TensorIOMode::kINPUT) {
      mInputNames.push_back(tensor_name);
      mInputDims[tensor_name] = mEngine->getTensorShape(tensor_name.c_str());;;
    } else {
      mOutputNames.push_back(tensor_name);
      mOutputDims[tensor_name] = mEngine->getTensorShape(tensor_name.c_str());;
    }
  }

  if (mBatchSize == 0) {
    mBatchSize = mEngine->getProfileShape(mInputNames[0].c_str(), 0, OptProfileSelector::kMAX).d[0];
  }
  return true;
}


void TRTEngine::setupInput(const std::string& input_name, const Dims& dims, bool host_buf, void* device_ptr) {
  // final shape = opt shape
  const std::string name = input_name.empty() ? mInputNames[0] : input_name;
  LOG(INFO) << "-------- setup input for name: " << name << "--------";

  auto final_shape = mInputDims.at(input_name);
  final_shape.d[0] = mBatchSize;

  mInputH = final_shape.d[2];
  mInputW = final_shape.d[3];

  mContext->setInputShape(name.c_str(), final_shape);
  LOG(INFO) << "model input '" << name << "', with shape: " << final_shape;

  if (device_ptr) {
    mContext->setTensorAddress(name.c_str(), device_ptr);
  } else  {
    mBufManager.initBuffer(getBufName(name), volume(final_shape) * sizeof(float), host_buf);
    mContext->setTensorAddress(name.c_str(),
                              mBufManager.getBuffer(getBufName(name), BUFFER_TYPE::DEVICE));
  }
}


void TRTEngine::setupOutput(const std::string& output_name, const Dims& dims,
                                   bool host_buf, void* device_ptr) {
  // final shape = opt shape
  LOG(INFO) << "-------- setup output for name: " << output_name << "--------\n";

  auto final_shape = mOutputDims.at(output_name);
  final_shape.d[0] = mBatchSize;
  LOG(INFO) << "model output '" << output_name << "', with shape: " << final_shape;
  if(device_ptr)  {
    mContext->setTensorAddress(output_name.c_str(), device_ptr);
  } else {
    mBufManager.initBuffer(getBufName(output_name), volume(final_shape) * sizeof(float), host_buf);
    mContext->setTensorAddress(output_name.c_str(),
                             mBufManager.getBuffer(getBufName(output_name), BUFFER_TYPE::DEVICE));
  }

}


bool TRTEngine::syncMemory(bool input, bool host2device, const cudaStream_t& stream) {
  std::vector<std::string>* names = nullptr;
  if (input) {
    names = &mInputNames;
  } else {
    names = &mOutputNames;
  }

  for (const auto& name : *names) {
    if (host2device) {  // host2device
      mBufManager.copyHostToDevice(getBufName(name), stream);
    } else {  // device2host
      mBufManager.copyDeviceToHost(getBufName(name), stream);
    }
  }
  return true;
}

nvinfer1::Dims TRTEngine::getBindingDims(bool is_input, const std::string& name) {
  const std::map<std::string, nvinfer1::Dims>* binding_set = nullptr;
  if (is_input) {
    binding_set = &mInputDims;
  } else {
    binding_set = &mOutputDims;
  }
  if (binding_set->count(name) > 0) {
    auto ret = binding_set->at(name);
    ret.d[0] = mBatchSize;
    return ret;
  } else {
    throw std::runtime_error("not output name: " + name);
  }
}


bool TRTEngine::infer(const cudaStream_t& stream) {
  if (!mContext->enqueueV3(stream)) {
    LOG(ERROR) << "enqueue failed!";
  };
  syncMemory(false, false, stream);
  return true;
}

}  // namespace nvocdr
