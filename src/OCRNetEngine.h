#ifndef __NVOCDR_OCRN_HEADER__
#define __NVOCDR_OCRN_HEADER__
#include <string>
#include "MemManager.h"
#include "TRTEngine.h"
#include "SimpleTokenizer.h"
#include "OCRNetBaseEngineVirtual.h"

#define OCRNET_OUTPUT_ID "output_id"
#define OCRNET_OUTPUT_PROB "output_prob"

using namespace nvinfer1;
namespace nvocdr
{




class OCRNetEngine: public OCRNetBaseEngine
{
    public:
        OCRNetEngine(const std::string& engine_path, const std::string& dict_path, const bool upside_down, const DecodeMode decode_mode);
        ~OCRNetEngine();

        bool initTRTBuffer(BufferManager& buffer_mgr);
        int32_t getMaxBatchSize() {return mEngine->getMaxBatchSize(); };
        bool setInputShape(const Dims& input_shape);
        bool setInputDeviceBuffer(DeviceBuffer& device_buffer, const int index);
        bool setOutputDeviceBuffer(DeviceBuffer& device_buffer, const int index);
        bool infer(BufferManager& buffer_mgr, std::vector<std::pair<std::string, float>>& de_texts,
                   const cudaStream_t& stream = 0);
        int mTRTOutputBufferIndex;

    private:
        std::unique_ptr<TRTEngine> mEngine;
        std::vector<std::string> mDict;
        bool mUDFlag;
        DecodeMode mDecodeMode;

};
}
#endif