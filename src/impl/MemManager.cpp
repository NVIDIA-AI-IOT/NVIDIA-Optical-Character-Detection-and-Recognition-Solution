#include <iostream>
#include "MemManager.h"

namespace nvocdr
{
void BufferManager::initBuffer(const std::string &name, const size_t& size, bool device, bool host) {
    // todo(shuohan) check if double init
    if (device && !mDeviceBuffer.count(name)) {
        mDeviceBuffer[name].resize(size);
        std::cerr << "init device buffer for '" << name << "' with bytes: " << size << "\n"; 
        mNumBytes[name] = size;
    }
    if (host && !mHostBuffer.count(name)) {
        mHostBuffer[name].resize(size);
        std::cerr << "init host buffer for '" << name << "' with bytes: " << size << "\n"; 
    }
}

void* BufferManager::getBuffer(const std::string & name, bool on_host) {
    if (on_host) {
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
void BufferManager::copyDeviceToHost(const std::string &name, const cudaStream_t& stream, bool async) {
    if (mHostBuffer.count(name) && mDeviceBuffer.count(name)) {
        checkCudaErrors(cudaMemcpyAsync(getBuffer(name, true), getBuffer(name, false), mNumBytes[name], cudaMemcpyDeviceToHost, stream));
    }
}
void BufferManager::copyHostToDevice(const std::string &name, const cudaStream_t& stream, bool async) {
    if (mHostBuffer.count(name) && mDeviceBuffer.count(name)) {
        checkCudaErrors(cudaMemcpyAsync(getBuffer(name, false), getBuffer(name, true), mNumBytes[name], cudaMemcpyHostToDevice, stream));
    }
}

size_t  BufferManager::getBufferSize(const std::string& name) {
    if (mNumBytes.count(name)) {
        return mNumBytes[name];
    }   
    return 0;
}
} // namespace nvocdr
