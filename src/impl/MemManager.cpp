#include <iostream>
#include "MemManager.h"
#include <glog/logging.h>

namespace nvocdr
{
void BufferManager::initBuffer(const std::string &name, const size_t& size, bool host_buf) {
    // todo(shuohan) check if double init
    if (!mDeviceBuffer.count(name)) {
        mDeviceBuffer[name].resize(size);
        LOG(INFO) << "init device buffer for '" << name << "' with bytes: " << size << "\n"; 
        mNumBytes[name] = size;
    }
    if (host_buf && !mHostBuffer.count(name)) {
        mHostBuffer[name].resize(size);
        LOG(INFO) << "init host buffer for '" << name << "' with bytes: " << size << "\n"; 
    }
}

void* BufferManager::getBuffer(const std::string & name, BUFFER_TYPE buf_type) {
    if (buf_type == BUFFER_TYPE::HOST) {
        if (mHostBuffer.count(name)) {
            return mHostBuffer.at(name).data();
        } else {
            throw std::runtime_error("can not get host buf " + name);
        }
    } else {
        if (mDeviceBuffer.count(name)) {
            return mDeviceBuffer.at(name).data();
        } else {
            throw std::runtime_error("can not get device buf " + name);
        }
    }
}
void BufferManager::copyDeviceToHost(const std::string &name, const cudaStream_t& stream) {
    if (mHostBuffer.count(name) && mDeviceBuffer.count(name)) {
        checkCudaErrors(cudaMemcpyAsync(getBuffer(name, BUFFER_TYPE::HOST), getBuffer(name, BUFFER_TYPE::DEVICE), mNumBytes[name], cudaMemcpyDeviceToHost, stream));
    }
}
void BufferManager::copyHostToDevice(const std::string &name, const cudaStream_t& stream) {
    if (mHostBuffer.count(name) && mDeviceBuffer.count(name)) {
        checkCudaErrors(cudaMemcpyAsync(getBuffer(name, BUFFER_TYPE::DEVICE), getBuffer(name, BUFFER_TYPE::HOST), mNumBytes[name], cudaMemcpyHostToDevice, stream));
    }
}

size_t  BufferManager::getBufferSize(const std::string& name) {
    if (mNumBytes.count(name)) {
        return mNumBytes[name];
    }   
    return 0;
}

} // namespace nvocdr
