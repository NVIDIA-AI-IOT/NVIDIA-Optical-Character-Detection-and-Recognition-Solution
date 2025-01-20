#ifndef __NVOCDR_OCRBASE_HEADER__
#define __NVOCDR_OCRBASE_HEADER__
#include <vector>
#include <memory>
#include "MemManager.h"
#include "TRTEngine.h"

using namespace nvinfer1;
namespace nvocdr
{

enum DecodeMode
{
    CTC,
    Attention,
    Transformer
};


class OCRNetBaseEngine
{
    public:
        virtual ~OCRNetBaseEngine(){};

        virtual bool initTRTBuffer(BufferManager& buffer_mgr) = 0;
        virtual int32_t getMaxBatchSize() = 0;
        virtual bool setInputShape(const Dims& input_shape) = 0;
        virtual bool infer(BufferManager& buffer_mgr, std::vector<std::pair<std::string, float>>& de_texts, const cudaStream_t& stream) = 0;
        // virtual std::pair<std::string, float> textDecode() = 0;
        int mTRTInputBufferIndex = 0;

    private:

        std::vector<std::string> mDict;

};
}
#endif