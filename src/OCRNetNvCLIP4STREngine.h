#ifndef __NVOCDR_OCRCLIP4STR_HEADER__
#define __NVOCDR_OCRCLIP4STR_HEADER__
#include <string>
#include "MemManager.h"
#include "TRTEngine.h"
#include "SimpleTokenizer.h"
#include "OCRNetBaseEngineVirtual.h"

using namespace nvinfer1;
namespace nvocdr
{


class OCRNetNvCLIP4STREngine : public OCRNetBaseEngine
{
public:
    OCRNetNvCLIP4STREngine(const std::string& vis_engine_path, const std::string& dict_path,
                     const bool upside_down, const std::string& text_engine_path="", bool only_alnum=true, bool only_lowercase=true, const std::string& vocab_file="", const int vocab_size=32000);
    ~OCRNetNvCLIP4STREngine();

    bool initTRTBuffer(BufferManager& buffer_mgr);
    int32_t getMaxBatchSize() {return mVisEngine->getMaxBatchSize(); };
    bool setInputShape(const Dims& input_shape);
    bool infer(BufferManager& buffer_mgr, std::vector<std::pair<std::string, float>>& de_texts, const cudaStream_t& stream = 0);
    std::pair<std::string, float> textDecode( const std::vector<float>& output_prob, const int batch_idx, const int context_len, const int charset_len);

private:
    std::unique_ptr<TRTEngine> mVisEngine;
    std::unique_ptr<TRTEngine> mTextEngine;
    std::unique_ptr<SimpleTokenizer> mTokenizer;
    std::vector<std::string> mDict;
    bool mUDFlag;
    DecodeMode mDecodeMode;
    // int mTRTInputBufferIndex;
    int mVisTRTOutputDecodeProbsBufferIndex;
    int mVisTRTOutputImgFeatureBufferIndex;
    int mVisTRTOutputContextBufferIndex;
    int mTextTRTInputTextTokenBufferIndex;
    int mTextTRTOutputBufferIndex;
    std::string mVisualInputName = "imgs";
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
};


}
#endif