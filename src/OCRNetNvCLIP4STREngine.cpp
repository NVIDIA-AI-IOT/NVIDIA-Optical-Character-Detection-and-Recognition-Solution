#include <iostream>
#include <fstream>
#include <algorithm>
#include "OCRNetNvCLIP4STREngine.h"
#include "kernel.h"

using namespace nvocdr;

OCRNetNvCLIP4STREngine::OCRNetNvCLIP4STREngine(const std::string& vis_engine_path, const std::string& dict_path, const bool upside_down=0, 
                            const std::string& text_engine_path, bool only_alnum, bool only_lowercase, const std::string& vocab_file, const int vocab_size)   
{
    // Init vis engine
    mVisEngine = std::move(std::unique_ptr<TRTEngine>(new TRTEngine(vis_engine_path)));
    // init text engine of CLIP4STR
    mTextEngine = std::move(std::unique_ptr<TRTEngine>(new TRTEngine(text_engine_path)));
    // vocab file
    mTokenizer = std::move(std::unique_ptr<SimpleTokenizer>(new SimpleTokenizer(vocab_file, vocab_size)));
    // Init Dict
    std::ifstream dict_file(dict_path);
    if (dict_file.good() == false)
    {
        std::cerr << "[ERROR] Error reading OCRNet dict file " << dict_path << std::endl;
    }
    mDict.clear();

    mDict.emplace_back("[E]");
    while(!dict_file.eof())
    {
        std::string ch;
        if (getline(dict_file, ch))
        {
            mDict.emplace_back(ch);
        }
    }

    mDict.emplace_back("[B]");
    mDict.emplace_back("[P]");

    mUDFlag = upside_down;
    mOnlyAlNum = only_alnum;
    mOnlyLowerCase = only_lowercase;
}

OCRNetNvCLIP4STREngine::~OCRNetNvCLIP4STREngine()
{
    mVisEngine.reset(nullptr);
    mTextEngine.reset(nullptr);
    mTokenizer.reset(nullptr);
}


bool
OCRNetNvCLIP4STREngine::initTRTBuffer(BufferManager& buffer_mgr)
{
    // Init visual branch trt input gpu buffer
    mTRTInputBufferIndex = buffer_mgr.initDeviceBuffer(mVisEngine->getMaxInputBufferSize(), sizeof(float));
    mVisEngine->setInputBuffer(buffer_mgr.mDeviceBuffer[mTRTInputBufferIndex].data());
    mTextEngine->setInputBatchSizebyName(mTextInTokenName, mVisEngine->getMaxBatchSize());
    // init visual branch trt ouput gpu buffer
    mVisTRTOutputImgFeatureBufferIndex = buffer_mgr.initDeviceBuffer(mVisEngine->getMaxTrtIoTensorSizeByName(mVisualOutImgFeaturebName), mVisEngine->getTrtIoTensorDtypeSizeByName(mVisualOutImgFeaturebName));
    mVisEngine->setOutputBufferByName(buffer_mgr.mDeviceBuffer[mVisTRTOutputImgFeatureBufferIndex].data(), mVisualOutImgFeaturebName);
    mVisTRTOutputDecodeProbsBufferIndex = buffer_mgr.initDeviceBuffer(mVisEngine->getMaxTrtIoTensorSizeByName(mVisualOutDecodeProbName), mVisEngine->getTrtIoTensorDtypeSizeByName(mVisualOutDecodeProbName));
    mVisEngine->setOutputBufferByName(buffer_mgr.mDeviceBuffer[mVisTRTOutputDecodeProbsBufferIndex].data(), mVisualOutDecodeProbName);
    mVisTRTOutputContextBufferIndex = buffer_mgr.initDeviceBuffer(mVisEngine->getMaxTrtIoTensorSizeByName(mVisualOutContextName), mVisEngine->getTrtIoTensorDtypeSizeByName(mVisualOutContextName));
    mVisEngine->setOutputBufferByName(buffer_mgr.mDeviceBuffer[mVisTRTOutputContextBufferIndex].data(), mVisualOutContextName);
    // init text branch trt in gpu buffers
    mTextTRTInputTextTokenBufferIndex = buffer_mgr.initDeviceBuffer(mTextEngine->getMaxTrtIoTensorSizeByName(mTextInTokenName), mTextEngine->getTrtIoTensorDtypeSizeByName(mTextInTokenName));
    mTextEngine->setInputBufferbyName(buffer_mgr.mDeviceBuffer[mTextTRTInputTextTokenBufferIndex].data(), mTextInTokenName);
    mTextEngine->setInputBufferbyName(buffer_mgr.mDeviceBuffer[mVisTRTOutputContextBufferIndex].data(), mVisualOutContextName);
    mTextEngine->setInputBufferbyName(buffer_mgr.mDeviceBuffer[mVisTRTOutputImgFeatureBufferIndex].data(), mVisualOutImgFeaturebName);
    mTextEngine->setInputBatchSizebyName(mTextInTokenName, mTextEngine->getMaxBatchSize());
    mTextEngine->setInputBatchSizebyName(mVisualOutContextName, mTextEngine->getMaxBatchSize());
    mTextEngine->setInputBatchSizebyName(mVisualOutImgFeaturebName, mTextEngine->getMaxBatchSize());
    // init text branch trt out gpu buffer
    mTextTRTOutputBufferIndex = buffer_mgr.initDeviceBuffer(mTextEngine->getMaxTrtIoTensorSizeByName(mTextOutLogitName), mTextEngine->getTrtIoTensorDtypeSizeByName(mTextOutLogitName));
    mTextEngine->setOutputBufferByName(buffer_mgr.mDeviceBuffer[mTextTRTOutputBufferIndex].data(), mTextOutLogitName);

    return true;
}

bool
OCRNetNvCLIP4STREngine::setInputShape(const Dims& input_shape)
{
    int maxBatchSize =  input_shape.d[0];
    mVisEngine->setInputBatchSizebyName(mVisualInputName, maxBatchSize);
    return 0;
}

bool
OCRNetNvCLIP4STREngine::infer(BufferManager& buffer_mgr, std::vector<std::pair<std::string, float>>& de_texts, const cudaStream_t& stream)
{
    int batch_size = 0;
    std::vector<std::pair<std::string, float>> temp_de_texts;
    mVisEngine->infer(stream);

    Dims visOutputDecodeProbShape = mVisEngine->getExactOutputShape(mVisualOutDecodeProbName);
    batch_size = visOutputDecodeProbShape.d[0];
    int context_max_length = visOutputDecodeProbShape.d[1];
    int charset_len = visOutputDecodeProbShape.d[2];
    
    // Get visual branch output
    std::vector<float> output_prob(buffer_mgr.mDeviceBuffer[mVisTRTOutputDecodeProbsBufferIndex].size());
    cudaMemcpyAsync(output_prob.data(), buffer_mgr.mDeviceBuffer[mVisTRTOutputDecodeProbsBufferIndex].data(),
                    buffer_mgr.mDeviceBuffer[mVisTRTOutputDecodeProbsBufferIndex].nbBytes(), cudaMemcpyDeviceToHost, stream);
    // CLIP4STR decode visual branch output
    std::vector<std::vector<int>> all_text_tokens;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx)
    {
        std::pair<std::string, float> de_text_prob = textDecode(output_prob, batch_idx, context_max_length, charset_len);
        // batch_captions.emplace_back(de_text);
        std::vector<int> text_tokens = mTokenizer->encode(de_text_prob.first);
        text_tokens.insert(text_tokens.begin(), mTokenizer->getStartTextToken());
        text_tokens.push_back(mTokenizer->getEndTextToken());

        std::vector<int> truncated_text_tokens(mMaxContextLen,0);
        if (text_tokens.size() > mMaxContextLen)
        {
            std::copy(text_tokens.begin(), text_tokens.begin() + mMaxContextLen, truncated_text_tokens.begin());
            truncated_text_tokens.back() = mTokenizer->getEndTextToken();
        }
        else
        {
            std::copy(text_tokens.begin(), text_tokens.end(), truncated_text_tokens.begin());
        }
        all_text_tokens.emplace_back(truncated_text_tokens);
    }

    // text branch inference
    cudaMemcpyAsync(buffer_mgr.mDeviceBuffer[mTextTRTInputTextTokenBufferIndex].data(), all_text_tokens.data(),
                    batch_size*mMaxContextLen*sizeof(int), cudaMemcpyHostToDevice, stream);
    mTextEngine->infer(stream);
    Dims textOutputDecodeProbShape = mTextEngine->getExactOutputShape(mTextOutLogitName);
    
    std::vector<float> text_output_prob(volume(textOutputDecodeProbShape));
    cudaMemcpyAsync(text_output_prob.data(), buffer_mgr.mDeviceBuffer[mTextTRTOutputBufferIndex].data(),
                    text_output_prob.size()*sizeof(float), cudaMemcpyDeviceToHost, stream);
                    // buffer_mgr.mDeviceBuffer[mTextTRTOutputBufferIndex].nbBytes(), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for (int i=0; i<batch_size; i++) {
        std::pair<std::string, float> de_text_prob = textDecode(text_output_prob, i, context_max_length, charset_len);
        temp_de_texts.emplace_back(de_text_prob);
    }
    

    int stride = batch_size / 2;
    int total_cnt = stride;
    if (mUDFlag)
    {
        for(int idx = 0; idx < total_cnt; idx += 1)
        {
            if (temp_de_texts[idx + stride].second > temp_de_texts[idx].second) 
            {
                de_texts.emplace_back(temp_de_texts[idx + stride]);
            }
            else
            {
                de_texts.emplace_back(temp_de_texts[idx]);
            }
        }
    }
    else
    {
        for(auto temp_text: temp_de_texts)
            de_texts.emplace_back(temp_text);
    }
    return 0;
}

std::pair<std::string, float> OCRNetNvCLIP4STREngine::textDecode( const std::vector<float>& output_prob, const int batch_idx, const int context_len, const int charset_len)
{
    std::string de_text = "";
    float prob = 1.0;
    for (int context_id = 0; context_id < context_len; ++context_id)
    {
        int batch_context_row_start = batch_idx * context_len * charset_len + context_id * charset_len;
        auto max_iter = std::max_element(output_prob.begin() + batch_context_row_start, output_prob.begin() + batch_context_row_start + charset_len);
        prob *= *max_iter;
        int id = std::distance(output_prob.begin() + batch_context_row_start, max_iter);
            if (mDict[id] == "[E]")
        {
            break;
        }
        if (mOnlyAlNum && !std::regex_match(mDict[id], std::regex("[a-zA-Z0-9]")))
        {
            continue;
        }
        if (mOnlyLowerCase)
        {
            std::string tmpCaption = mDict[id];
            for (char& c : tmpCaption) {
                c = static_cast<char>(std::tolower(c));
            }
            de_text += tmpCaption;
        }
        else
        {
            de_text += mDict[id];
        }
        
    }
    return std::make_pair(de_text, prob);
}
