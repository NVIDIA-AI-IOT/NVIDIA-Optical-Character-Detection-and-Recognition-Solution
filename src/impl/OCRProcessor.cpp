#include <algorithm>
#include <fstream>
#include <iostream>

#include <glog/logging.h>

#include "MemManager.h"
#include "OCRProcessor.h"

namespace nvocdr {
OCRProcessor::OCRProcessor(const char name[], const nvOCRParam& param) : BaseProcessor<nvOCRParam>(param) {
    if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CTC) {
        std::string model_file(mParam.model_file);
        mEngines[CTC_MODEL].reset(new TRTEngine(CTC_MODEL, model_file, mParam.batch_size));
    }
}
size_t OCRProcessor::getBatchSize() {
  if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CTC) {
    return mEngines[CTC_MODEL]->getBatchSize();
    initCTC();
    mEngines[CTC_MODEL]->postInit();
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_ATTN) {

  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
  } else {
    LOG(INFO) << "[ERROR] Unsupported decode mode";
  }
};

bool OCRProcessor::init() {
  mDict.clear();
  if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CTC) {
    mEngines[CTC_MODEL]->initEngine();
    initCTC();
    mEngines[CTC_MODEL]->postInit();
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_ATTN) {

  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
  } else {
    LOG(INFO) << "[ERROR] Unsupported decode mode";
  }
  return true;
}
std::string OCRProcessor::getInputBufName() {
  if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CTC) {
    return mEngines[CTC_MODEL]->getBufName(OCRNET_INPUT);
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_ATTN) {
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
  } else {
    LOG(INFO) << "[ERROR] Unsupported decode mode";
    throw std::runtime_error("Unsupported decode mode");
  }
};
cv::Size OCRProcessor::getInputHW() {
  if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CTC) {
    auto in_dims = mEngines[CTC_MODEL]->getInputDims(OCRNET_INPUT);
    return {in_dims.d[3], in_dims.d[2]};
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_ATTN) {
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
  } else {
    LOG(INFO) << "[ERROR] Unsupported decode mode";
  }
}

void OCRProcessor::initCTC() {
    mDict.emplace_back("CTCBlank");
    loadDict();

  mEngines[CTC_MODEL]->setupInput(OCRNET_INPUT, {}, true);
  mEngines[CTC_MODEL]->setupOutput(OCRNET_OUTPUT_PROB, {}, true);
  mEngines[CTC_MODEL]->setupOutput(OCRNET_OUTPUT_ID, {}, true);

  mOutputCharLength = mEngines[CTC_MODEL]->getOutputDims(OCRNET_OUTPUT_ID).d[1];
  LOG(INFO) << "decode length: " << mOutputCharLength;

  std::stringstream ss;
  for (auto const& c : mDict) {
    ss << c << ", ";
  }
  LOG(INFO) << "dict: " << ss.str();
}

void OCRProcessor::initATTN() {
    mDict.emplace_back("[GO]");
    mDict.emplace_back("[s]");
    loadDict();
}

void OCRProcessor::initCLIP() {
    mDict.emplace_back("[E]");
    loadDict();
    mDict.emplace_back("[B]");
    mDict.emplace_back("[P]");
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
    decodeATTNOrCLIP(text, idx, "[s]");
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
    decodeATTNOrCLIP(text, idx, "[E]");
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

void OCRProcessor::decodeATTNOrCLIP(Text* const text, size_t idx, const std::string& ending) {

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
}
}  // namespace nvocdr
