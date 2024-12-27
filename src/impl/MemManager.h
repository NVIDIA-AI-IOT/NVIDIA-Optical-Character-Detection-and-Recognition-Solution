#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <memory>
#include <vector>
#include <unordered_map>
#include <map>

namespace nvocdr {

    #ifndef checkKernelErrors
#    define checkKernelErrors(expr)                                                               \
        do                                                                                        \
        {                                                                                         \
            expr;                                                                                 \
                                                                                                  \
            cudaError_t __err = cudaGetLastError();                                               \
            if (__err != cudaSuccess)                                                             \
            {                                                                                     \
                printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err)); \
                abort();                                                                          \
            }                                                                                     \
        }                                                                                         \
        while (0)
#endif

// CUDA Runtime error messages
static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

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
class GenericBuffer
{
public:
    //!
    //! \brief Construct an empty buffer.
    //!
    // GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
    GenericBuffer()
        : mSize(0)
        , mCapacity(0)
        , mItemSize(1)
        , mBuffer(nullptr)
    {
    }
    GenericBuffer(const size_t item_size)
        : mSize(0)
        , mCapacity(0)
        , mItemSize(item_size)
        , mBuffer(nullptr)
    {
    }

    //!
    //! \brief Construct a buffer with the specified allocation size in bytes.
    //!
    // GenericBuffer(size_t size, nvinfer1::DataType type)
    GenericBuffer(size_t size, size_t item_size)
        : mSize(size)
        , mItemSize(item_size)
        , mCapacity(size)
    {
        if (!allocFn(&mBuffer, this->nbBytes()))
        {
            throw std::bad_alloc();
        }
    }

    GenericBuffer(GenericBuffer&& buf)
        : mSize(buf.mSize)
        , mCapacity(buf.mCapacity)
        , mItemSize(buf.mItemSize)
        , mBuffer(buf.mBuffer)
    {
        buf.mSize = 0;
        buf.mCapacity = 0;
        buf.mItemSize = 0;
        buf.mBuffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf)
        {
            freeFn(mBuffer);
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mItemSize= buf.mItemSize;
            mBuffer = buf.mBuffer;
            // Reset buf.
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    void* data()
    {
        return mBuffer;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    const void* data() const
    {
        return mBuffer;
    }

    //!
    //! \brief Returns the size (in number of elements) of the buffer.
    //!
    size_t size() const
    {
        return mSize;
    }

    //!
    //! \brief Returns the size (in bytes) of the buffer.
    //!
    size_t nbBytes() const
    {
        return this->size() * mItemSize;
    }

    //!
    //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
    //!
    void resize(size_t newSize)
    {
        mSize = newSize;
        if (mCapacity < newSize)
        {
            freeFn(mBuffer);
            if (!allocFn(&mBuffer, this->nbBytes()))
            {
                throw std::bad_alloc{};
            }
            mCapacity = newSize;
        }
    }

    ~GenericBuffer()
    {
        freeFn(mBuffer);
    }

private:
    size_t mSize{0}, mCapacity{0};
    size_t mItemSize;
    void* mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

class DeviceAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        return cudaMalloc(ptr, size) == cudaSuccess;
    }
};

class DeviceFree
{
public:
    void operator()(void* ptr) const
    {
        cudaFree(ptr);
    }
};

class HostAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }
};

class HostFree
{
public:
    void operator()(void* ptr) const
    {
        free(ptr);
    }
};

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

//Helper class that control all the mem used in the lib
//Version 0 
enum BUFFER_TYPE {
    DEVICE,
    HOST
};
class BufferManager
{
    public:
        void initBuffer(const std::string& name, const size_t& size, bool host_buf = true);
        void* getBuffer(const std::string & name, BUFFER_TYPE buf_type);
        bool checkBufferExist(const std::string & name) {return mDeviceBuffer.count(name) > 0;};
        size_t getBufferSize(const std::string& name);
        void copyDeviceToHost(const std::string &name, const cudaStream_t& stream);
        void copyHostToDevice(const std::string &name, const cudaStream_t& stream);

        static BufferManager& Instance() {
          static BufferManager singleton;
          return singleton;
        }
        BufferManager(const BufferManager &) = delete;
        BufferManager & operator = (const BufferManager &) = delete;

    private:
      BufferManager() {}
      ~BufferManager() {}
      std::map<std::string, DeviceBuffer> mDeviceBuffer;
      std::map<std::string, HostBuffer> mHostBuffer;
      std::map<std::string, size_t> mNumBytes;
};

}

