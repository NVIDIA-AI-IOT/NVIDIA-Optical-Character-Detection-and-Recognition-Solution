#ifndef __NVOCDR_OCRN_HEADER__
#define __NVOCDR_OCRN_HEADER__
#include <string>
#include "MemManager.h"
#include "TRTEngine.h"
#include "SimpleTokenizer.h"

#define OCRNET_OUTPUT_ID "output_id"
#define OCRNET_OUTPUT_PROB "output_prob"

using namespace nvinfer1;
namespace nvocdr
{

enum DecodeMode
{
    CTC,
    Attention,
    Transformer
};


class OCRNetEngine
{
    public:
        OCRNetEngine(const std::string& engine_path, const std::string& dict_path,
                     const bool upside_down, const DecodeMode decode_mode, const std::string& text_engine_path="", bool only_alnum=true, bool only_lowercase=true, const std::string& vocab_file="", const int vocab_size=32000);
        ~OCRNetEngine();

        bool initTRTBuffer(BufferManager& buffer_mgr);
        int32_t getMaxBatchSize() {return mEngine->getMaxBatchSize(); };
        bool setInputShape(const Dims& input_shape);
        bool setInputDeviceBuffer(DeviceBuffer& device_buffer, const int index);
        bool setOutputDeviceBuffer(DeviceBuffer& device_buffer, const int index);
        bool infer(BufferManager& buffer_mgr, std::vector<std::pair<std::string, float>>& de_texts,
                   const cudaStream_t& stream = 0);
        std::pair<std::string, float> clip4strDecode( const std::vector<float>& output_prob, const int batch_idx, const int context_len, const int charset_len);
        int mTRTInputBufferIndex;
        int mTRTOutputBufferIndex;
        // CLIP4STR buffer
        int mVisTRTOutputDecodeProbsBufferIndex;
        int mVisTRTOutputImgFeatureBufferIndex;
        int mVisTRTOutputContextBufferIndex;

        int mTextTRTInputTextTokenBufferIndex;
        int mTextTRTOutputBufferIndex;
    private:
        std::unique_ptr<TRTEngine> mEngine;
        std::vector<std::string> mDict;
        bool mUDFlag;
        DecodeMode mDecodeMode;

        // CLIP4STR
        std::unique_ptr<TRTEngine> mTextEngine;
        std::string mVisualOutDecodeProbName = "visual_decode_probs";
        std::string mVisualOutContextName = "tgt_in";
        std::string mVisualOutImgFeaturebName = "img_feature";
        std::string mTextInTokenName = "text_token";
        std::string mTextOutLogitName = "logits";
        int mVocabSize = 32000;
        int mMaxContextLen = 16;
        // only output alpha and digit
        bool mOnlyAlNum = true;
        // only output lower case alpha
        bool mOnlyLowerCase = true;
        std::unique_ptr<SimpleTokenizer> mTokenizer;
};
}
#endif