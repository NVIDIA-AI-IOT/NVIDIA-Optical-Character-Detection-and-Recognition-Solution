#include <iostream>
#include <fstream>
#include "OCRNetEngine.h"
#include "kernel.h"

using namespace nvocdr;


OCRNetEngine::OCRNetEngine(const std::string& engine_path, const std::string& dict_path, const bool upside_down, const DecodeMode decode_mode)
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
    else
    {
        mDict.emplace_back("[GO]");
        mDict.emplace_back("[s]");
    };

    while(!dict_file.eof())
    {
        std::string ch;
        if (getline(dict_file, ch))
        {
            mDict.emplace_back(ch);
        }
    }

    mUDFlag = upside_down;

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

    // Init trt output gpu buffer
    mTRTOutputBufferIndex = buffer_mgr.initDeviceBuffer(mEngine->getMaxOutputBufferSize(), sizeof(float));
    mEngine->setOutputBuffer(buffer_mgr.mDeviceBuffer[mTRTOutputBufferIndex].data());

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
    
    mEngine->infer(stream);

    // CPU Decode:
    Dims output_prob_shape = mEngine->getExactOutputShape(OCRNET_OUTPUT_PROB);
    Dims output_id_shape = mEngine->getExactOutputShape(OCRNET_OUTPUT_ID);
    int batch_size = output_prob_shape.d[0];
    int output_len = output_prob_shape.d[1];

    std::vector<float> output_prob(volume(output_prob_shape));
    std::vector<int> output_id(volume(output_id_shape));
    cudaMemcpyAsync(output_prob.data(), mEngine->getOutputAddr(OCRNET_OUTPUT_PROB),
                    output_prob.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(output_id.data(), mEngine->getOutputAddr(OCRNET_OUTPUT_ID),
                    output_id.size() * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<std::pair<std::string, float>> temp_de_texts;
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
    else
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