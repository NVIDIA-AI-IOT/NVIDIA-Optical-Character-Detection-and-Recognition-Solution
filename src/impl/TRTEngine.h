#pragma once

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "nvocdr.h"
#include "MemManager.h"

namespace nvocdr
{
using namespace nvinfer1;

size_t volume(const Dims& dim);

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

inline std::ostream& operator << (std::ostream& o, const Dims& dims) {
    o << '[';
    for(size_t i = 0; i < dims.nbDims; ++i) {
        o << (i > 0 ? "," : "") << dims.d[i];
    }
    o << ']';
    return o;
}

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, InferDeleter>;

// TRT Engine wrapper on TRT 8.5 API for version 0:
// - Single input and multiple output
// - Dynamic batch
// - FP32 input & output
template<typename Param>
class TRTEngine
{
    public:
        TRTEngine(const char name[], const Param& param) : mName(name), mParam(param) {};

        bool init();
        bool initEngine();
        virtual bool customInit() = 0;

        // Do TRT-based inference with processed input and get output
        bool infer(const cudaStream_t& stream = 0);
        bool syncMemory(bool input, bool host2device, const cudaStream_t& stream);
        std::string getBufName(const std::string &name) {return mName + "_" + name; }

        // Manager will only give a buffer with max size and let TRT Engine to use the buffer
        // void setInputBuffer(void* buffer);

        // void setInputShape(const Dims shape);

        // size_t getMaxInputBufferSize();

        // Dims getExactInputShape() {return mExactInputShape;};

        // Dims getMaxInputShape() {return mMaxInputShape;};

        // size_t getMaxBatchSize() {return mMaxBatchSize;};

        // size_t getMaxOutputBufferSize();

        // void setOutputBuffer(void* buffer);

        // const void* getOutputAddr(std::string output_name);

        // Dims getOutputShapeByName(std::string output_name);

        // Get the size of memeory will be used by TensorRT engine internally
        // int getInterMemSize();
        inline size_t getInputH () { return mInputH; };
        inline size_t getInputW () { return mInputW; };
        inline size_t getBatchSize() { return mBatchSize; }

    protected:
        void setupInput(const std::string &input_name, const Dims& dims, bool host_buf = false);
        void setupOutput(const std::string &ouput_name, const Dims& dims, bool host_buf = false);



        Param mParam;
        std::string mName;
        BufferManager &mBufManager = BufferManager::Instance();

    private:
        TRTUniquePtr<IRuntime> mRuntime;
        TRTUniquePtr<ICudaEngine> mEngine;
        TRTUniquePtr<IExecutionContext> mContext;

        size_t mBatchSize = 1U;
        size_t mInputH;
        size_t mInputW;
        // std::string mInputName;
        // Dims mMaxInputShape; // with bs
        // size_t mMaxBatchSize;
        // Dims mExactInputShape; // with bs
        // std::vector<std::string> mOutputNames;
        // std::vector<Dims> mMaxOutputShapes; // with bs

        // std::unordered_map<std::string, Dims> mInputs; // from engine model
        // std::unordered_map<std::string, Dims> mOutputs; // from engine model
        std::vector<std::string> mInputNames;
        std::vector<std::string> mOutputNames;
        // std::map<std::string, std::string> mInputBufNames;
        // std::map<std::string, std::string> mOutputBufNames;
};

using OCRTRTEngine = TRTEngine<nvOCRParam>;
using OCDTRTEngine = TRTEngine<nvOCDParam>;

// template<> class TRTEngine<nvOCRParam>;


}