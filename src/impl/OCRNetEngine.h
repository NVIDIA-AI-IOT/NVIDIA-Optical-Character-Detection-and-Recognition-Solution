#pragma once
#include <string>
#include <vector>
#include "MemManager.h"
#include "TRTEngine.h"
#include "nvocdr.h"


namespace nvocdr
{
    

constexpr char OCR_PREFIX[] = "OCR";
// todo (shuohanc) add  input name 
constexpr char OCRNET_OUTPUT_ID[] = "output_id";
constexpr char OCRNET_OUTPUT_PROB[] = "output_prob";

class OCRNetEngine: public OCRTRTEngine
{
    // using OCRTRTEngine::TRTEngine;
    public:
        bool customInit() final;
        // OCRNetEngine() = default;
        OCRNetEngine(const char name[], const nvOCRParam& param) : OCRTRTEngine(name, param) { };
        

        // bool initTRTBuffer(BufferManager& buffer_mgr);
        // int32_t getMaxBatchSize() {return mEngine->getMaxBatchSize(); };
        // bool setInputShape(const Dims& input_shape);
        // bool setInputDeviceBuffer(DeviceBuffer& device_buffer, const int index);
        // bool setOutputDeviceBuffer(DeviceBuffer& device_buffer, const int index);
        // bool infer(const cudaStream_t& stream = 0);
        // nvOCDROutputBlob* const getOutput();
        // size_t getBatchSize() {return mEngine->getBatchSize();};
        // the batch may greater than actual size, use decode cnt to control
        void decode(const nvOCDROutput* output, size_t decode_cnt, const cudaStream_t& stream);

    private:
        // nvOCRParam mParam;
        // size_t mTRTInputBufferIndex;
        // size_t mTRTOutputBufferIndex;
        // std::unique_ptr<TRTEngine> mEngine;

        std::vector<std::string> mDict;
        // std::vector<nvOCDROutputBlob> mOutputs;

        // bool mUDFlag;
        // OCRNetDecode mDecodeMode;

        // int mDecodeOutputBufferIndex;
};
}