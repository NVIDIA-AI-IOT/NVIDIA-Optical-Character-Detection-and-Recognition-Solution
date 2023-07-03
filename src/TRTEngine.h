#ifndef __NVOCDR_TRT_HEADER__
#define __NVOCDR_TRT_HEADER__
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include "NvInfer.h"

using namespace nvinfer1;
namespace nvocdr
{

size_t volume(const nvinfer1::Dims& dim);

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, InferDeleter>;

// TRT Engine wrapper on TRT 8.5 API for version 0:
// - Single input and multiple output
// - Dynamic batch
// - FP32 input & output
class TRTEngine
{
    public:
        // Do deserialize
        TRTEngine(const std::string engine_path);

        // Destroy TRT related resources
        ~TRTEngine();

        // Do TRT-based inference with processed input and get output
        bool infer(const cudaStream_t& stream = 0);

        // Manager will only give a buffer with max size and let TRT Engine to use the buffer
        void setInputBuffer(void* buffer);

        void setInputShape(const Dims shape);

        size_t getMaxInputBufferSize();

        Dims getExactInputShape() {return mExactInputShape;};

        Dims getMaxInputShape() {return mMaxInputShape;};

        size_t getMaxBatchSize() {return mMaxBatchSize;};

        size_t getMaxOutputBufferSize();

        void setOutputBuffer(void* buffer);

        const void* getOutputAddr(std::string output_name);

        Dims getExactOutputShape(std::string output_name);

        // Get the size of memeory will be used by TensorRT engine internally
        // int getInterMemSize();

    private:
        TRTUniquePtr<IRuntime> mRuntime;
        TRTUniquePtr<ICudaEngine> mEngine;
        TRTUniquePtr<IExecutionContext> mContext;
        std::string mInputName;
        Dims mMaxInputShape; // with bs
        size_t mMaxBatchSize;
        Dims mExactInputShape; // with bs
        std::vector<std::string> mOutputNames;
        std::vector<Dims> mMaxOutputShapes; // with bs
        std::unordered_map<std::string, Dims> mExactOutputShapes; // with bs
};

}

#endif