#ifndef __NVOCDR_MEM_HEADER__
#define __NVOCDR_MEM_HEADER__
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace nvocdr
{
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
class PairBuffer
{
public:
    DeviceBuffer deviceBuffer;
    HostBuffer hostBuffer;
};

//Helper class that control all the mem used in the lib
//Version 0 
class BufferManager
{
    public:
        BufferManager();

        int initDeviceBuffer(const size_t size, const size_t item_size);
        int initHostBuffer(const size_t size, const size_t item_size);

        std::vector<DeviceBuffer> mDeviceBuffer;
        std::vector<HostBuffer> mHostBuffer;
};

}

#endif