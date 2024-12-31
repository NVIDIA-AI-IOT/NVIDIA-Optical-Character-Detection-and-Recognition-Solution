#include <glog/logging.h>
#include <iostream>
#include <string>
#include <cstddef>
#include "macro.h"

#include <cuda_runtime_api.h>
// #include <cuda_runtime.h>

#include "MemManager.h"

namespace nvocdr {
void BufferManager::initBuffer(const std::string& name, const size_t& size, bool host_buf, uint8_t *data) {
  // todo(shuohan) check if double init
  if (mDeviceBuffer.count(name) == 0) {
    mDeviceBuffer[name].resize(size);
    LOG(INFO) << "init device buffer for '" << name << "' with bytes: " << size << "\n";
    mNumBytes[name] = size;
  } else {
    LOG(WARNING) << "device buffer already exist: " << name;
  }
  if (host_buf && mHostBuffer.count(name) == 0) {
    mHostBuffer[name].resize(size);
    LOG(INFO) << "init host buffer for '" << name << "' with bytes: " << size << "\n";
    if (data) {
      LOG(INFO) << "copy data to host buffer";
      memcpy(mHostBuffer[name].data(), data, size);
    }
  }
}

void BufferManager::initBuffer2D(const std::string& name, size_t c, size_t h, size_t w, size_t elem_size, bool host_buf, uint8_t *data) {
  if (mDeviceBuffer.count(name) == 0) {
    mDeviceBuffer[name].resize(c, h, w, elem_size, &mBuf2DPitch[name]);
    LOG(INFO) << "init device 2D buffer for '" << name;
    auto &dim = mBuf2DDims[name];
    dim.nbDims = 4;
    dim.d[0] = 1;
    dim.d[1] = c;
    dim.d[2] = h;
    dim.d[3] = w;
    mBuf2D.insert(name);
  } else {
    LOG(WARNING) << "device buffer already exist: " << name;
  } 
  if (host_buf && mHostBuffer.count(name) == 0) {
    mHostBuffer[name].resize(c * h * w * elem_size);
    LOG(INFO) << "init host buffer for '" << name << "' with bytes: " << c * h * w * elem_size << "\n";
    if (data) {
      LOG(INFO) << "copy data to host buffer";
      memcpy(mHostBuffer[name].data(), data, c * h * w * elem_size);
      
    }
  }
}

void* BufferManager::getBuffer(const std::string& name, BUFFER_TYPE buf_type) {
  if (buf_type == BUFFER_TYPE::HOST) {
    if (mHostBuffer.count(name) > 0) {
      return mHostBuffer.at(name).data();
    } else {
      throw std::runtime_error("can not get host buf " + name);
    }
  } else {
    if (mDeviceBuffer.count(name) > 0) {
      return mDeviceBuffer.at(name).data();
    } else {
      throw std::runtime_error("can not get device buf " + name);
    }
  }
}
void BufferManager::copyDeviceToHost(const std::string& name, const cudaStream_t& stream) {
  if (mHostBuffer.count(name) > 0 && mDeviceBuffer.count(name) > 0) {

    if(mBuf2D.count(name) > 0) {
      auto &dim = mBuf2DDims[name];

      checkCudaErrors(cudaMemcpy2D(getBuffer(name, BUFFER_TYPE::HOST), 
        dim.d[3] * dim.d[1], getBuffer(name, BUFFER_TYPE::DEVICE), mBuf2DPitch[name], dim.d[3] * dim.d[1], dim.d[2], cudaMemcpyDeviceToHost));
    } else {
       checkCudaErrors(cudaMemcpyAsync(getBuffer(name, BUFFER_TYPE::HOST),
                                    getBuffer(name, BUFFER_TYPE::DEVICE), mNumBytes[name],
                                    cudaMemcpyDeviceToHost, stream));
    }
  }
}
void BufferManager::copyHostToDevice(const std::string& name, const cudaStream_t& stream) {
  if (mHostBuffer.count(name) > 0 && mDeviceBuffer.count(name) > 0) {
    if(mBuf2D.count(name) > 0) {
      auto &dim = mBuf2DDims[name];
      checkCudaErrors(cudaMemcpy2D(getBuffer(name, BUFFER_TYPE::DEVICE), 
        mBuf2DPitch[name], getBuffer(name, BUFFER_TYPE::HOST), dim.d[3] * dim.d[1], dim.d[3] * dim.d[1], dim.d[2], cudaMemcpyHostToDevice));
    } else {
      checkCudaErrors(cudaMemcpyAsync(getBuffer(name, BUFFER_TYPE::DEVICE),
                                    getBuffer(name, BUFFER_TYPE::HOST), mNumBytes[name],
                                    cudaMemcpyHostToDevice, stream));
    }
  }
}

nvinfer1::Dims BufferManager::getBuf2DDim(const std::string& name) {
  if (mBuf2D.count(name) == 0) {
    throw std::runtime_error("only 2D buffer can get dim");
  } else {
    return mBuf2DDims[name];
  }
}
size_t  BufferManager::getBuf2DPitch(const std::string& name) {
  if (mBuf2D.count(name) == 0) {
    throw std::runtime_error("only 2D buffer can get pitch");
  } else {
    return mBuf2DPitch[name];
  }
}


void BufferManager::releaseAllBuffers() {
  mDeviceBuffer.clear();
  mHostBuffer.clear();
}


}  // namespace nvocdr
