#pragma once

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include <set>

#include <NvInfer.h>

namespace nvocdr {
// NOLINTBEGIN

inline size_t volume(const nvinfer1::Dims& dim) {
  size_t data_size = 1;
  for (size_t i = 0; i < dim.nbDims; i++) {
    data_size *= dim.d[i];
  }
  return data_size;
}

//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
//!          size is the amount of memory in bytes to allocate.
//!          The boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer address. It must work with nullptr input.
//!
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer {
 public:
  //!
  //! \brief Construct an empty buffer.
  //!
  GenericBuffer() : mSize(0), mCapacity(0), mItemSize(1), mBuffer(nullptr) {}

  void* data() { return mBuffer; }

  const void* data() const { return mBuffer; }

  //!
  //! \brief Returns the size (in number of elements) of the buffer.
  //!
  size_t size() const { return mSize; }

  //!
  //! \brief Returns the size (in bytes) of the buffer.
  //!
  size_t nbBytes() const { return this->size() * mItemSize; }

  //!
  //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
  //!
  void resize(size_t newSize) {
    mSize = newSize;
    if (mCapacity < newSize) {
      freeFn(mBuffer);
      if (!allocFn(&mBuffer, this->nbBytes())) {
        throw std::bad_alloc{};
      }
      mCapacity = newSize;
    }
    
  }
  void resize(size_t c, size_t h, size_t w, size_t elem_size, size_t *pitch) {
    auto vol_size = c * h * w;
    if (mBuffer) {
      freeFn(mBuffer);
    }
    if (!allocFn(&mBuffer, c, h, w, elem_size, pitch)) {
        throw std::bad_alloc{};
    }
  }
  ~GenericBuffer() { freeFn(mBuffer); }

 private:
  size_t mSize{0};
  size_t mCapacity{0};
  size_t mItemSize;
  void* mBuffer = nullptr;
  AllocFunc allocFn;
  FreeFunc freeFn;
  // int mPitch;
};

// {

// }

class DeviceAllocator {
 public:
  bool operator()(void** ptr, size_t size) const { return cudaMalloc(ptr, size) == cudaSuccess; }
  bool operator()(void** ptr, size_t c, size_t h, size_t w, size_t elem_size, size_t *pitch) const { return cudaMallocPitch(ptr, pitch, w * c * elem_size, h) == cudaSuccess; }
};

class DeviceFree {
 public:
  void operator()(void* ptr) const { cudaFree(ptr); }
};

class HostAllocator {
 public:
  bool operator()(void** ptr, size_t size) const {
    *ptr = malloc(size);
    return *ptr != nullptr;
  }
};

class HostFree {
 public:
  void operator()(void* ptr) const { free(ptr); }
};
// NOLINTEND

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

//!
//! \brief  The ManagedBuffer class groups together a pair of corresponding device and host buffers.
//!
// class PairBuffer
// {
// public:
//     DeviceBuffer deviceBuffer;
//     HostBuffer hostBuffer;
// };

enum BUFFER_TYPE : uint8_t 
{ DEVICE, HOST };
class BufferManager {
 public:
  void initBuffer(const std::string& name, const size_t& size, bool host_buf = true, uint8_t *data = nullptr);
  void initBuffer2D(const std::string& name, size_t c, size_t h, size_t w, size_t elem_size, bool host_buf = true, uint8_t *data = nullptr);
  void* getBuffer(const std::string& name, BUFFER_TYPE buf_type);
  bool checkBufferExist(const std::string& name) { return mDeviceBuffer.count(name) > 0; };
  // size_t getBufferSize(const std::string& name);
  void copyDeviceToHost(const std::string& name, const cudaStream_t& stream);
  void copyHostToDevice(const std::string& name, const cudaStream_t& stream);
  nvinfer1::Dims getBuf2DDim(const std::string& name);
  size_t getBuf2DPitch(const std::string& name);
  void releaseAllBuffers();

  static BufferManager& Instance() {
    static BufferManager singleton;
    return singleton;
  }
  BufferManager(const BufferManager&) = delete;
  BufferManager& operator=(const BufferManager&) = delete;

 private:
  BufferManager() = default;
  ~BufferManager() = default;
  std::map<std::string, DeviceBuffer> mDeviceBuffer;
  std::map<std::string, HostBuffer> mHostBuffer;
  std::map<std::string, size_t> mNumBytes;

  std::map<std::string, nvinfer1::Dims> mBuf2DDims;
  std::map<std::string, size_t> mBuf2DPitch;
  std::set<std::string> mBuf2D;
};

}  // namespace nvocdr
