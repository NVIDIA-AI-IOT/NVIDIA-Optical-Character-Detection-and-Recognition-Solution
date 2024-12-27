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
inline size_t volume(const nvinfer1::Dims& dim) {
  size_t data_size = 1;
  for (size_t i = 0; i < dim.nbDims; i++) {
    data_size *= dim.d[i];
  }
  return data_size;
}

class Logger : public ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    // suppress info-level messages
    if (severity <= Severity::kERROR){
      std::cout << msg << '\n';}
  }
} logger;

template <typename Param>
bool TRTEngine<Param>::init() {
  initEngine();
  customInit();
  postInit();
  return true;
}

template <typename Param>
bool TRTEngine<Param>::postInit() {
  for (auto const& name : mOutputNames) {
    if (!mBufManager.checkBufferExist(getBufName(name))) {
      LOG(WARNING) << "buffer for '" << name << "' not allocated, allocate to max";
      setupOutput(name, {}, false);  // remove in model export
    }
  }
  return true;
}

template <typename Param>
bool TRTEngine<Param>::initEngine() {

  initLibNvInferPlugins(reinterpret_cast<void*>(&logger), "");
  mRuntime = std::move(TRTUniquePtr<IRuntime>(createInferRuntime(logger)));
  //load binary engine:
  LOG(INFO) << "load engine from:" << mParam.model_file;
  std::ifstream engineFile(mParam.model_file, std::ios::binary);
  if (!engineFile.good()) {
    LOG(ERROR) << "Error reading serialized TensorRT engine: " << mParam.model_file;
  }
  engineFile.seekg(0, std::ifstream::end);
  const int64_t fsize = engineFile.tellg();
  engineFile.seekg(0, std::ifstream::beg);

  std::vector<char> engineBlob(fsize);
  engineFile.read(engineBlob.data(), fsize);
  if (!engineFile.good()) {
    LOG(INFO) << "Error reading serialized TensorRT engine: " << mParam.model_file;
    throw std::runtime_error("Error reading serialized TensorRT engine");
  }

  mEngine = std::move(TRTUniquePtr<ICudaEngine>(
      mRuntime->deserializeCudaEngine(engineBlob.data(), engineBlob.size())));
  mContext = std::move(TRTUniquePtr<IExecutionContext>(mEngine->createExecutionContext()));

  size_t engine_max_batch_size = 0;
  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const std::string tensor_name(mEngine->getIOTensorName(i));

    if (mEngine->getTensorIOMode(tensor_name.c_str()) == TensorIOMode::kINPUT) {
      mInputNames.push_back(tensor_name);
      engine_max_batch_size =
          mEngine->getProfileShape(tensor_name.c_str(), 0, OptProfileSelector::kMAX).d[0];
    } else {
      mOutputNames.push_back(tensor_name);
    }
  }

  if (mParam.batch_size == 0) {
    mBatchSize = engine_max_batch_size;
  }
  return true;
}

template <typename Param>
void TRTEngine<Param>::setupInput(const std::string& input_name, const Dims& dims, bool host_buf) {
  // final shape = opt shape
  const std::string name = input_name.empty() ? mInputNames[0] : input_name;
  LOG(INFO) << "-------- setup input for name: " << name << "--------";

  auto final_shape = mEngine->getProfileShape(name.c_str(), 0, OptProfileSelector::kOPT);

  if (dims.nbDims != 0) {  // if dims provided
                           // todo(shuohan) add check here if input dims bigger than max;
  } else {                 // if no dims provided, use max batch size + opt
    auto const max_shape = mEngine->getProfileShape(name.c_str(), 0, OptProfileSelector::kMAX);
    final_shape.d[0] = mBatchSize;
  }

  mInputH = final_shape.d[2];
  mInputW = final_shape.d[3];

  mContext->setInputShape(name.c_str(), final_shape);
  mInputDims[name] = final_shape;
  LOG(INFO) << "model input '" << name << "', with shape: " << final_shape;
  mBufManager.initBuffer(getBufName(name), volume(final_shape) * sizeof(float), host_buf);
  mContext->setTensorAddress(name.c_str(),
                             mBufManager.getBuffer(getBufName(name), BUFFER_TYPE::DEVICE));
}

template <typename Param>
void TRTEngine<Param>::setupOutput(const std::string& output_name, const Dims& dims,
                                   bool host_buf) {
  // final shape = opt shape
  mOutputNames.push_back(output_name);
  LOG(INFO) << "-------- setup output for name: " << output_name << "--------\n";

  auto final_shape = mEngine->getTensorShape(output_name.c_str());
  if (dims.nbDims != 0) {  // if dims provided
                           // todo(shuohan) add check here if input dims bigger than profile max;
                           // auto const max_shape = mEngine->getTensorShape(output_name.c_str());
  } else {                 // if no dims provided, use max batch size + opt
    final_shape.d[0] = mBatchSize;
  }
  // mContext->setOutputShape(output_name.c_str(), final_shape);
  LOG(INFO) << "model output '" << output_name << "', with shape: " << final_shape;

  // todo(shuohanc) hardcode for float for now, cause all our model has 32bit output
  mBufManager.initBuffer(getBufName(output_name), volume(final_shape) * sizeof(float), host_buf);
  mOutputDims[output_name] = final_shape;
  mContext->setTensorAddress(output_name.c_str(),
                             mBufManager.getBuffer(getBufName(output_name), BUFFER_TYPE::DEVICE));
}

template <typename Param>
bool TRTEngine<Param>::syncMemory(bool input, bool host2device, const cudaStream_t& stream) {
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
              //   std::cout<<"device to host " << name ;
      mBufManager.copyDeviceToHost(getBufName(name), stream);
    }
  }
}
template <typename Param>
nvinfer1::Dims TRTEngine<Param>::getOutputDims(const std::string& name) {
  if (mOutputDims.count(name) > 0) {
    return mOutputDims[name];
  } else {
    throw std::runtime_error("not output name: " + name);
  }
}

template <typename Param>
bool TRTEngine<Param>::infer(const cudaStream_t& stream) {
  if (!mContext->enqueueV3(stream)) {
    LOG(ERROR) << "enqueue failed!";
  };
  return true;
}

template class TRTEngine<nvOCRParam>;
template class TRTEngine<nvOCDParam>;

}  // namespace nvocdr
