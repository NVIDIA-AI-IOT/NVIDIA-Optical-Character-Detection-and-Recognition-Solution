#include <algorithm>
#include <fstream>
#include <iostream>

#include <glog/logging.h>

#include "MemManager.h"
#include "OCRProcessor.h"

namespace nvocdr {
OCRProcessor::OCRProcessor(const nvOCRParam& param) : BaseProcessor<nvOCRParam>(param) {
    if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CTC) {
        std::string model_file(mParam.model_file);
        mEngines[CTC_MODEL].reset(new TRTEngine(CTC_MODEL, model_file, mParam.batch_size));
    } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
        std::string model_file_packed(mParam.model_file); 
        std::string visual_model_model_file = model_file_packed.substr(0, model_file_packed.find(','));
        std::string text_model_model_file = model_file_packed.substr(model_file_packed.find(',') + 1);
        mEngines[CLIP_VISUAL_MODEL].reset(new TRTEngine(CLIP_VISUAL_MODEL, visual_model_model_file, mParam.batch_size));
        mEngines[CLIP_TEXT_MODEL].reset(new TRTEngine(CLIP_TEXT_MODEL, text_model_model_file, mParam.batch_size));
    }
}
size_t OCRProcessor::getBatchSize() {
  if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CTC) {
    return mEngines[CTC_MODEL]->getBatchSize();
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_ATTN) {
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
    return mEngines[CLIP_VISUAL_MODEL]->getBatchSize(); // text and visual has same batch size
  } else {
    LOG(INFO) << "[ERROR] Unsupported decode mode";
  }
};

bool OCRProcessor::init() {
  mDict.clear();
  if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CTC) {
    initCTC();
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_ATTN) {
    LOG(INFO) << "[ERROR] Unsupported model type";
    throw std::runtime_error("Unsupported ATTN");
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
    initCLIP();
  } else {
    LOG(INFO) << "[ERROR] Unsupported model type";
    throw std::runtime_error("Unsupported model type");
  }

  std::stringstream ss;
  for (auto const& c : mDict) {
    ss << c << ", ";
  }
  LOG(INFO) << "dict: " << ss.str();

  return true;
}
std::string OCRProcessor::getInputBufName() {
  if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CTC) {
    return mEngines[CTC_MODEL]->getBufName(OCRNET_INPUT);
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_ATTN) {
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
    return mEngines[CLIP_VISUAL_MODEL]->getBufName(CLIP_VISUAL_INPUT);
  } else {
    LOG(INFO) << "[ERROR] Unsupported decode mode";
    throw std::runtime_error("Unsupported decode mode");
  }
};
cv::Size OCRProcessor::getInputHW() {
  if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CTC) {
    auto in_dims = mEngines[CTC_MODEL]->getBindingDims(true, OCRNET_INPUT);
    return {in_dims.d[3], in_dims.d[2]};
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_ATTN) {
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
    auto in_dims = mEngines[CLIP_VISUAL_MODEL]->getBindingDims(true, CLIP_VISUAL_INPUT);
    return {in_dims.d[3], in_dims.d[2]};
  } else {
    LOG(INFO) << "[ERROR] Unsupported decode mode";
  }
}

void OCRProcessor::initCTC() {
  // init engine
  mEngines[CTC_MODEL]->initEngine();
  mEngines[CTC_MODEL]->setupInput(OCRNET_INPUT, {}, true);
  mEngines[CTC_MODEL]->setupOutput(OCRNET_OUTPUT_PROB, {}, true);
  mEngines[CTC_MODEL]->setupOutput(OCRNET_OUTPUT_ID, {}, true);
  mEngines[CTC_MODEL]->postInit();
  mOutputCharLength = mEngines[CTC_MODEL]->getBindingDims(false, OCRNET_OUTPUT_ID).d[1];
  LOG(INFO) << "decode length: " << mOutputCharLength;

  // init dict
  mDict.emplace_back("CTCBlank");
  loadDict();
}

void OCRProcessor::initATTN() {
    mDict.emplace_back("[GO]");
    mDict.emplace_back("[s]");
    loadDict();
}

void OCRProcessor::initCLIP() {
    // setup engine for visual 
    mEngines[CLIP_VISUAL_MODEL]->initEngine();
    mEngines[CLIP_VISUAL_MODEL]->setupInput(CLIP_VISUAL_INPUT, {}, true);
    mEngines[CLIP_VISUAL_MODEL]->setupOutput(CLIP_VISUAL_OUTPUT_PROBS, {}, true);

    auto clip_feature_dim = mEngines[CLIP_VISUAL_MODEL]->getBindingDims(false, CLIP_VISUAL_OUTPUT_FEATURE);
    auto clip_id_dim = mEngines[CLIP_VISUAL_MODEL]->getBindingDims(false, CLIP_VISUAL_OUTPUT_ID);
    auto clip_feature_buf_name = mEngines[CLIP_VISUAL_MODEL]->getBufName(CLIP_VISUAL_OUTPUT_FEATURE);
    auto clip_id_buf_name = mEngines[CLIP_VISUAL_MODEL]->getBufName(CLIP_VISUAL_OUTPUT_ID);

    mBufManager.initBuffer(clip_feature_buf_name, volume(clip_feature_dim) * sizeof(float), false);
    mBufManager.initBuffer(clip_id_buf_name, volume(clip_id_dim) * sizeof(int), false);

    auto* feature_dev_ptr = mBufManager.getBuffer(clip_feature_buf_name, DEVICE);
    auto* id_dev_ptr = mBufManager.getBuffer(clip_id_buf_name, DEVICE);
    mEngines[CLIP_VISUAL_MODEL]->setupOutput(CLIP_VISUAL_OUTPUT_FEATURE, {}, false, feature_dev_ptr);
    mEngines[CLIP_VISUAL_MODEL]->setupOutput(CLIP_VISUAL_OUTPUT_ID, {}, false, id_dev_ptr);

    // setup engine for text
    mEngines[CLIP_TEXT_MODEL]->initEngine();
    mEngines[CLIP_TEXT_MODEL]->setupInput(CLIP_TEXT_INPUT_TEXT_TOKEN, {}, true);
    mEngines[CLIP_TEXT_MODEL]->setupInput(CLIP_TEXT_INPUT_FEATURE, {}, false, feature_dev_ptr);
    mEngines[CLIP_TEXT_MODEL]->setupInput(CLIP_TEXT_INPUT_ID, {}, false, id_dev_ptr);
    mEngines[CLIP_TEXT_MODEL]->setupOutput(CLIP_TEXT_OUTPUT_LOGITS, {}, true);


    mEngines[CLIP_VISUAL_MODEL]->postInit();
    mEngines[CLIP_TEXT_MODEL]->postInit();


    mDict.emplace_back("[E]");
    loadDict();
    mDict.emplace_back("[B]");
    mDict.emplace_back("[P]");

    auto batch_size = getBatchSize();
    auto clip_prob_dim = mEngines[CLIP_VISUAL_MODEL]->getBindingDims(false, CLIP_VISUAL_OUTPUT_PROBS);
    mOutputCharLength = clip_prob_dim.d[1];
    mClipCharTypeSize = clip_prob_dim.d[2];

    mTmpText.resize(batch_size);
    for(size_t i = 0; i < batch_size;++i ) {
        mTmpText[i].resize(mOutputCharLength);
    }
    LOG(INFO) << "init CLIP visual temp space " << mOutputCharLength << "x" << mClipCharTypeSize;
    // LOG(INFO) << "clip dict: " << CLIP_DICT;
}


void OCRProcessor::loadDict() {
  LOG(INFO) << "reading dict from: " << mParam.dict;
  if (std::string(mParam.dict) == "default") {
    LOG(INFO) << "using default 0-9a-z as dictionary";
    for (size_t i = 0; i <= 9; ++i) {
      mDict.emplace_back(std::to_string(i));
    }

    for (size_t i = 0; i < 26; i++) {
      mDict.emplace_back(1, i + 'a');
    }
  } else {
    std::ifstream dict_file(mParam.dict);

    if (!dict_file.good()) {
      LOG(INFO) << "[ERROR] Error reading OCRNet dict file " << mParam.dict << std::endl;
    }
    while (!dict_file.eof()) {
      std::string ch;
      if (getline(dict_file, ch)) {
        mDict.emplace_back(ch);
      }
    }
  }
}

void OCRProcessor::decode(Text* const text, size_t idx) {
  if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CTC) {
    decodeCTC(text, idx);
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_ATTN) {
    // decodeATTNOrCLIP(text, idx, "[s]");
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
    // decodeATTNOrCLIP(text, idx, "[E]");
  }
}

void OCRProcessor::decodeCTC(Text* const text, size_t idx) {

  auto &ctc_engine = mEngines.at(CTC_MODEL);
  int offset = idx * mOutputCharLength;
  float* prob = reinterpret_cast<float*>(
                    mBufManager.getBuffer(ctc_engine->getBufName(OCRNET_OUTPUT_PROB), BUFFER_TYPE::HOST)) +
                offset;
  // for trt above 8.6, this node must be casted to int32
  int32_t* cls = reinterpret_cast<int32_t*>(
                     mBufManager.getBuffer(ctc_engine->getBufName(OCRNET_OUTPUT_ID), BUFFER_TYPE::HOST)) +
                 offset;

  std::string ret;
  float score = 1.F;
  for (size_t i = 0, j = 0; i < mOutputCharLength && j < mOutputCharLength; i++) {
    while (j < mOutputCharLength && (cls[j] == 0 || cls[j] == cls[i])){
      j++;
    }
    ret += mDict[cls[i]];
    score *= prob[i];
    i = j - 1;
  }

  // we may recognize multi times, use best
  if (score > text->conf) {
    memcpy(text->text, ret.c_str(), ret.length());
    text->text[ret.length()] = '\0';

    text->text_length = ret.length();
    text->conf = score;
  }
}

bool OCRProcessor::infer(bool sync_input, const cudaStream_t& stream) {
  if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
    // clip visual infer
    mEngines[CLIP_VISUAL_MODEL]->infer(stream);
    decodeCLIPVisual(stream);
    // tokenizer

    // clip text infer
    mEngines[CLIP_TEXT_MODEL]->infer(stream);
  } else {
    BaseProcessor<nvOCRParam>::infer(sync_input, stream);
  } 
  return true;  
};

// inline fi
void OCRProcessor::decodeCLIPVisual(const cudaStream_t& stream) {
  const auto prob_buf_name = mEngines[CLIP_VISUAL_MODEL]->getBufName(CLIP_VISUAL_OUTPUT_PROBS);
  mBufManager.copyDeviceToHost(prob_buf_name, stream);  
  auto* prob = static_cast<float*>(mBufManager.getBuffer(prob_buf_name, HOST));
  
  auto batch_size = getBatchSize();
  
  for(size_t i = 0; i < batch_size; ++i) {
    for(size_t j = 0; j < mOutputCharLength; ++j) {
        float max_prob = -1;
        size_t max_idx = 0;
        for(size_t t = 0; t < mClipCharTypeSize; ++t) {
          auto char_prob = prob[i * mOutputCharLength * mClipCharTypeSize + j * mClipCharTypeSize + t];
        //   LOG(INFO) << char_prob;
          if (char_prob > max_prob) {
            max_prob = char_prob;
            max_idx = t;
          }
        }
        mTmpText[i][j] = CLIP_CHAR[max_idx];
    }
    LOG(INFO) << std::string(mTmpText[i].begin(), mTmpText[i].end());
  }

}


// void OCRProcessor::decodeATTN(Text* const text, size_t idx, const std::string& ending) {

//   size_t offset = idx * mOutputCharLength;
//   float* prob = reinterpret_cast<float*>(
//                     mBufManager.getBuffer(getBufName(OCRNET_OUTPUT_PROB), BUFFER_TYPE::HOST)) +
//                 offset;
//   // for trt above 8.6, this node must be casted to int32
//   int32_t* cls = reinterpret_cast<int32_t*>(
//                      mBufManager.getBuffer(getBufName(OCRNET_OUTPUT_ID), BUFFER_TYPE::HOST)) +
//                  offset;

//   std::string ret;
//   float score = 1.F;
//   for (size_t i = 0; i < mOutputCharLength; i++) {
//     if (mDict[cls[i]] == ending) {
//       break;
//     }
//     ret += mDict[cls[i]];
//     score *= prob[i];
//   }
//   memcpy(text->text, ret.c_str(), ret.length());
//   text->text[ret.length()] = '\0';

//   text->text_length = ret.length();
//   text->conf = score;
// }
}  // namespace nvocdr
