#include <iostream>
#include <fstream>
#include <algorithm>
#include "OCRNetEngine.h"
#include "kernel.h"

using namespace nvocdr;


OCRNetEngine::OCRNetEngine(const std::string& engine_path, const std::string& dict_path, const bool upside_down=0, const DecodeMode decode_mode=Transformer, 
                            const std::string& text_engine_path, bool only_alnum, bool only_lowercase, const std::string& vocab_file, const int vocab_size)   
{
    // Init TRTEngine
    mEngine = std::move(std::unique_ptr<TRTEngine>(new TRTEngine(engine_path)));

    // Init Dict
    std::ifstream dict_file(dict_path);
    if (dict_file.good() == false)
    {
        std::cerr << "[ERROR] Error reading OCRNet dict file " << dict_path << std::endl;
    }
    mDict.clear();
    mDecodeMode = decode_mode;
    if (mDecodeMode == CTC)
    {
        mDict.emplace_back("CTCBlank");
    }
    else if (mDecodeMode == Attention)
    {
        mDict.emplace_back("[GO]");
        mDict.emplace_back("[s]");
    }
    else if (mDecodeMode == Transformer)
    {
        mDict.emplace_back("[E]");
    } 
    else
    {
        std::cerr << "[ERROR] Unsupported decode mode" << std::endl;
    }

    while(!dict_file.eof())
    {
        std::string ch;
        if (getline(dict_file, ch))
        {
            mDict.emplace_back(ch);
        }
    }

    if (mDecodeMode == Transformer)
    {
        mDict.emplace_back("[B]");
        mDict.emplace_back("[P]");
        // init text engine of CLIP4STR
        mTextEngine = std::move(std::unique_ptr<TRTEngine>(new TRTEngine(text_engine_path)));
        // vocab file
        mTokenizer.initTokenizer(vocab_file, vocab_size);
    }

    mUDFlag = upside_down;
    mOnlyAlNum = only_alnum;
    mOnlyLowerCase = only_lowercase;
}


OCRNetEngine::~OCRNetEngine()
{
    mEngine.reset(nullptr);
}


bool
OCRNetEngine::initTRTBuffer(BufferManager& buffer_mgr)
{
    // Init trt input gpu buffer
    mTRTInputBufferIndex = buffer_mgr.initDeviceBuffer(mEngine->getMaxInputBufferSize(), sizeof(float));
    mEngine->setInputBuffer(buffer_mgr.mDeviceBuffer[mTRTInputBufferIndex].data());
    if (mDecodeMode != Transformer)
    {
        // Init trt output gpu buffer
        mTRTOutputBufferIndex = buffer_mgr.initDeviceBuffer(mEngine->getMaxOutputBufferSize(), sizeof(float));
        mEngine->setOutputBuffer(buffer_mgr.mDeviceBuffer[mTRTOutputBufferIndex].data());
    }
    else
    {
        // init CLIP4STR visual branch trt ouput gpu buffer
        mVisTRTOutputImgFeatureBufferIndex = buffer_mgr.initDeviceBuffer(mEngine->getMaxTrtIoTensorSizeByName(mVisualOutImgFeaturebName), mEngine->getTrtIoTensorDtypeSizeByName(mVisualOutImgFeaturebName));
        mEngine->setOutputBufferByName(buffer_mgr.mDeviceBuffer[mVisTRTOutputImgFeatureBufferIndex].data(), mVisualOutImgFeaturebName);
        mVisTRTOutputDecodeProbsBufferIndex = buffer_mgr.initDeviceBuffer(mEngine->getMaxTrtIoTensorSizeByName(mVisualOutDecodeProbName), mEngine->getTrtIoTensorDtypeSizeByName(mVisualOutDecodeProbName));
        mEngine->setOutputBufferByName(buffer_mgr.mDeviceBuffer[mVisTRTOutputDecodeProbsBufferIndex].data(), mVisualOutDecodeProbName);
        mVisTRTOutputContextBufferIndex = buffer_mgr.initDeviceBuffer(mEngine->getMaxTrtIoTensorSizeByName(mVisualOutContextName), mEngine->getTrtIoTensorDtypeSizeByName(mVisualOutContextName));
        mEngine->setOutputBufferByName(buffer_mgr.mDeviceBuffer[mVisTRTOutputContextBufferIndex].data(), mVisualOutContextName);
        // init CLIP4STR text branch trt in gpu buffers
        mTextTRTInputTextTokenBufferIndex = buffer_mgr.initDeviceBuffer(mTextEngine->getMaxTrtIoTensorSizeByName(mTextInTokenName), mTextEngine->getTrtIoTensorDtypeSizeByName(mTextInTokenName));
        mTextEngine->setInputBufferbyName(buffer_mgr.mDeviceBuffer[mTextTRTInputTextTokenBufferIndex].data(), mTextInTokenName);
        mTextEngine->setInputBufferbyName(buffer_mgr.mDeviceBuffer[mVisTRTOutputContextBufferIndex].data(), mVisualOutContextName);
        mTextEngine->setInputBufferbyName(buffer_mgr.mDeviceBuffer[mVisTRTOutputImgFeatureBufferIndex].data(), mVisualOutImgFeaturebName);
        mTextEngine->setInputBatchSizebyName(mTextInTokenName, mTextEngine->getMaxBatchSize());
        mTextEngine->setInputBatchSizebyName(mVisualOutContextName, mTextEngine->getMaxBatchSize());
        mTextEngine->setInputBatchSizebyName(mVisualOutImgFeaturebName, mTextEngine->getMaxBatchSize());
        // init CLIP4STR text branch trt out gpu buffer
        mTextTRTOutputBufferIndex = buffer_mgr.initDeviceBuffer(mTextEngine->getMaxTrtIoTensorSizeByName(mTextOutLogitName), mTextEngine->getTrtIoTensorDtypeSizeByName(mTextOutLogitName));
        mTextEngine->setOutputBufferByName(buffer_mgr.mDeviceBuffer[mTextTRTOutputBufferIndex].data(), mTextOutLogitName);
    }
    

    return 0;
}


bool
OCRNetEngine::setInputShape(const Dims& input_shape)
{
    mEngine->setInputShape(input_shape);
    return 0;
}


bool
OCRNetEngine::setInputDeviceBuffer(DeviceBuffer& device_buffer, const int index)
{
    mTRTInputBufferIndex = index;
    mEngine->setInputBuffer(device_buffer.data());
    return 0;
}


bool
OCRNetEngine::setOutputDeviceBuffer(DeviceBuffer& device_buffer, const int index)
{
    mTRTOutputBufferIndex = index;
    mEngine->setOutputBuffer(device_buffer.data());
    return 0;
}


bool
OCRNetEngine::infer(BufferManager& buffer_mgr, std::vector<std::pair<std::string, float>>& de_texts, const cudaStream_t& stream)
{

    // Preprocess:
    // unsigned int item_cnt = volume(mEngine->getExactInputShape());
    // float mean = 127.5;
    // float scale = 0.00784313;
    // subscal(item_cnt, buffer_mgr.mDeviceBuffer[mTRTInputBufferIndex].data(), scale, mean, stream);
    int batch_size = 0;
    std::vector<std::pair<std::string, float>> temp_de_texts;
    mEngine->infer(stream);
    
    if (mDecodeMode != Transformer)
    {
        // CPU Decode:
        Dims output_prob_shape = mEngine->getExactOutputShape(OCRNET_OUTPUT_PROB);
        Dims output_id_shape = mEngine->getExactOutputShape(OCRNET_OUTPUT_ID);
        batch_size = output_prob_shape.d[0];
        int output_len = output_prob_shape.d[1];

        std::vector<float> output_prob(volume(output_prob_shape));
        std::vector<int> output_id(volume(output_id_shape));
        cudaMemcpyAsync(output_prob.data(), mEngine->getOutputAddr(OCRNET_OUTPUT_PROB),
                        output_prob.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(output_id.data(), mEngine->getOutputAddr(OCRNET_OUTPUT_ID),
                        output_id.size() * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);


        if (mDecodeMode == CTC)
        {
            for(int batch_idx = 0; batch_idx < batch_size; ++batch_idx)
            {
                int b_offset = batch_idx * output_len; 
                int prev = output_id[b_offset];
                std::vector<int> temp_seq_id = {prev};
                std::vector<float> temp_seq_prob = {output_prob[b_offset]};
                for(int i = 1 ; i < output_len; ++i)
                {
                    if (output_id[b_offset + i] != prev)
                    {
                        temp_seq_id.push_back(output_id[b_offset + i]);
                        temp_seq_prob.push_back(output_prob[b_offset + i]);
                        prev = output_id[b_offset + i];
                    }
                }
                std::string de_text = "";
                float prob = 1.0;
                for(size_t i = 0; i < temp_seq_id.size(); ++i)
                {
                    if (temp_seq_id[i] != 0)
                    {
                        if (temp_seq_id[i] <= static_cast<int>(mDict.size()) - 1)
                        {
                            de_text += mDict[temp_seq_id[i]];
                            prob *= temp_seq_prob[i];
                        }
                        else
                        {
                            std::cerr << "[ERROR] Character dict is not compatible with OCRNet TRT engine." << std::endl;
                        }
                    }
                }
                temp_de_texts.emplace_back(std::make_pair(de_text, prob));
            }
        }
        else if (mDecodeMode == Attention)
        {
            for(int batch_idx = 0; batch_idx < batch_size; ++batch_idx)
            {
                int b_offset = batch_idx * output_len;
                int stop_idx = 0;
                std::string de_text = "";
                float prob = 1.0;
                for(int i = 0; i < output_len; ++i)
                {
                    if (mDict[output_id[b_offset + i]] != "[s]")
                    {
                        de_text += mDict[output_id[b_offset + i]];
                        prob *= output_prob[b_offset + i];
                    }
                    else
                    {
                        break;
                    }
                }
                temp_de_texts.emplace_back(std::make_pair(de_text, prob));
            }
        }
    }
    
    else if (mDecodeMode == Transformer)
    {
        // CPU Decode:
        Dims visOutputDecodeProbShape = mEngine->getExactOutputShape(mVisualOutDecodeProbName);
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
            std::pair<std::string, float> de_text_prob = clip4strDecode(output_prob, batch_idx, context_max_length, charset_len);
            // batch_captions.emplace_back(de_text);
            std::vector<int> text_tokens = mTokenizer.encode(de_text_prob.first);
            text_tokens.insert(text_tokens.begin(), mTokenizer.getStartTextToken());
            text_tokens.push_back(mTokenizer.getEndTextToken());

            std::vector<int> truncated_text_tokens(mMaxContextLen,0);
            if (text_tokens.size() > mMaxContextLen)
            {
                std::copy(text_tokens.begin(), text_tokens.begin() + mMaxContextLen, truncated_text_tokens.begin());
                truncated_text_tokens.back() = mTokenizer.getEndTextToken();
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
                        buffer_mgr.mDeviceBuffer[mTextTRTOutputBufferIndex].nbBytes(), cudaMemcpyDeviceToHost, stream);
        for (int i=0; i<batch_size; i++) {
           std::pair<std::string, float> de_text_prob = clip4strDecode(text_output_prob, i, context_max_length, charset_len);
           temp_de_texts.emplace_back(de_text_prob);
        }
    }
    else
    {
        std::cerr << "[ERROR] Unsupported decode mode" << std::endl;
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


std::pair<std::string, float> OCRNetEngine::clip4strDecode( const std::vector<float>& output_prob, const int batch_idx, const int context_len, const int charset_len)
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
