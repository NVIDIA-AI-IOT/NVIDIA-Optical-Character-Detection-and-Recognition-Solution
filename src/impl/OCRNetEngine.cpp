#include <iostream>
#include <fstream>
#include <algorithm>
#include "OCRNetEngine.h"
#include "kernel.h"
#include "MemManager.h"

using namespace nvocdr;

bool OCRNetEngine::customInit()
{
    std::cerr << "reading dict from: " << mParam.dict_file << "\n";
    // Init Dict
    std::ifstream dict_file(mParam.dict_file);
    if (dict_file.good() == false)
    {
        std::cerr << "[ERROR] Error reading OCRNet dict file " << mParam.dict_file << std::endl;
    }
    mDict.clear();

    // mDecodeMode = mParam.mode;
    if (mParam.mode == CTC)
    {
        mDict.emplace_back("CTCBlank");
    }
    else if (mParam.mode == Attention)
    {
        mDict.emplace_back("[GO]");
        mDict.emplace_back("[s]");
    }
    else if (mParam.mode == CLIP)
    {
        mDict.emplace_back("[E]");
    }
    else
    {
        std::cerr << "[ERROR] Unsupported decode mode" << std::endl;
    }

    while (!dict_file.eof())
    {
        std::string ch;
        if (getline(dict_file, ch))
        {
            mDict.emplace_back(ch);
        }
    }

    if (mParam.mode == CLIP)
    {
        mDict.emplace_back("[B]");
        mDict.emplace_back("[P]");
    }

    setupInput(OCRNET_INPUT, {}, true);
    setupOutput(OCRNET_OUTPUT_PROB, {}, true);
    setupOutput(OCRNET_OUTPUT_ID, {}, true);
    setupOutput("798", {}, false); // remove in model export

    mOutputCharLength = getOutputDims(OCRNET_OUTPUT_ID).d[1];
    std::cout << "decode length: " << mOutputCharLength << "\n";

    for(size_t i = 0; i < mDict.size(); ++i) {
        std::cout << mDict[i] << " ";
    }
            std::cout << "\n";

    return true;
}

void OCRNetEngine::decode(Text *const text, size_t idx)
{
    if (mParam.mode == CTC) {
        decodeCTC(text, idx);
    } else if(mParam.mode == Attention) {

    } else if(mParam.mode == CLIP) {

    }
}

void OCRNetEngine::decodeCTC(Text *const text, size_t idx)
{

    int offset = idx * mOutputCharLength;
    float *prob = reinterpret_cast<float *>(mBufManager.getBuffer(getBufName(OCRNET_OUTPUT_PROB), true)) + offset;
    // for trt above 8.6, this node must be casted to int32
    int32_t *cls = reinterpret_cast<int32_t *>(mBufManager.getBuffer(getBufName(OCRNET_OUTPUT_ID), true)) + offset;

    std::string ret;
    float score = 1.F;
    for(size_t i = 0, j = 0; i < mOutputCharLength && j < mOutputCharLength; i++) {
        while(j < mOutputCharLength && (cls[j] == 0 || cls[j] == cls[i])) j++;
        // std::cout<< "cls: " << mDict[cls[i]] << " ";
        ret += mDict[cls[i]];
        score *= prob[i];
        i = j-1;
    }
    // std::cout<<"\n";
    memcpy(text->text, ret.c_str(), ret.length());
    text->text[ret.length()] = '\0';
    
    text->text_length = ret.length();
    text->conf = score;
}

// bool
// OCRNetEngine::infer(const cudaStream_t& stream)
// {

// Preprocess:
// unsigned int item_cnt = volume(mEngine->getExactInputShape());
// float mean = 127.5;
// float scale = 0.00784313;
// subscal(item_cnt, buffer_mgr.mDeviceBuffer[mTRTInputBufferIndex].data(), scale, mean, stream);

// mEngine->infer(stream);

// BufferManager::Instance().copyDeviceToHost(, stream);
// cudaStreamSynchronize(stream);
// CPU Decode:
// Dims output_prob_shape = mEngine->getOutputShapeByName(OCRNET_OUTPUT_PROB);
// Dims output_id_shape = mEngine->getOutputShapeByName(OCRNET_OUTPUT_ID);
// int batch_size = output_prob_shape.d[0];
// int output_len = output_prob_shape.d[1];

// std::vector<float> output_prob(volume(output_prob_shape));
// std::vector<int> output_id(volume(output_id_shape));

// cudaMemcpyAsync(output_prob.data(), mEngine->getOutputAddr(OCRNET_OUTPUT_PROB),
//                 output_prob.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
// cudaMemcpyAsync(output_id.data(), mEngine->getOutputAddr(OCRNET_OUTPUT_ID),
//                 output_id.size() * sizeof(int), cudaMemcpyDeviceToHost, stream);

// std::vector<std::pair<std::string, float>> temp_de_texts;
// if (mParam.mode == CTC)
// {
//     for(int batch_idx = 0; batch_idx < batch_size; ++batch_idx)
//     {
//         int b_offset = batch_idx * output_len;
//         int prev = output_id[b_offset];
//         std::vector<int> temp_seq_id = {prev};
//         std::vector<float> temp_seq_prob = {output_prob[b_offset]};
//         for(int i = 1 ; i < output_len; ++i)
//         {
//             if (output_id[b_offset + i] != prev)
//             {
//                 temp_seq_id.push_back(output_id[b_offset + i]);
//                 temp_seq_prob.push_back(output_prob[b_offset + i]);
//                 prev = output_id[b_offset + i];
//             }
//         }
//         std::string de_text = "";
//         float prob = 1.0;
//         for(size_t i = 0; i < temp_seq_id.size(); ++i)
//         {
//             if (temp_seq_id[i] != 0)
//             {
//                 if (temp_seq_id[i] <= static_cast<int>(mDict.size()) - 1)
//                 {
//                     de_text += mDict[temp_seq_id[i]];
//                     prob *= temp_seq_prob[i];
//                 }
//                 else
//                 {
//                     std::cerr << "[ERROR] Character dict is not compatible with OCRNet TRT engine." << std::endl;
//                 }
//             }
//         }
//         temp_de_texts.emplace_back(std::make_pair(de_text, prob));
//     }
// }
// else if (mParam.mode == Attention)
// {
//     for(int batch_idx = 0; batch_idx < batch_size; ++batch_idx)
//     {
//         int b_offset = batch_idx * output_len;
//         int stop_idx = 0;
//         std::string de_text = "";
//         float prob = 1.0;
//         for(int i = 0; i < output_len; ++i)
//         {
//             if (mDict[output_id[b_offset + i]] != "[s]")
//             {
//                 de_text += mDict[output_id[b_offset + i]];
//                 prob *= output_prob[b_offset + i];
//             }
//             else
//             {
//                 break;
//             }
//         }
//         temp_de_texts.emplace_back(std::make_pair(de_text, prob));
//     }
// }
// else if (mParam.mode == CLIP)
// {
//     for(int batch_idx = 0; batch_idx < batch_size; ++batch_idx)
//     {
//         int b_offset = batch_idx * output_len;
//         std::string de_text = "";
//         float prob = 1.0;

//         for(int i = 0; i < output_len; ++i)
//         {
//             if (mDict[output_id[b_offset + i]] == "[E]")
//             {
//                 break;
//             }
//             de_text += mDict[output_id[b_offset + i]];
//             prob *= output_prob[b_offset + i];
//         }

//         temp_de_texts.emplace_back(std::make_pair(de_text, prob));
//     }

// }
// else
// {
//     std::cerr << "[ERROR] Unsupported decode mode" << std::endl;
// }

// int stride = batch_size / 2;
// int total_cnt = stride;
// if (mUDFlag)
// {
//     for(int idx = 0; idx < total_cnt; idx += 1)
//     {
//         if (temp_de_texts[idx + stride].second > temp_de_texts[idx].second)
//         {
//             de_texts.emplace_back(temp_de_texts[idx + stride]);
//         }
//         else
//         {
//             de_texts.emplace_back(temp_de_texts[idx]);
//         }
//     }
// }
// else
// {
//     for(auto temp_text: temp_de_texts)
//         de_texts.emplace_back(temp_text);
// }
//     return true;
// }

