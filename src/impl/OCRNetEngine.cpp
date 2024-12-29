#include <algorithm>
#include <fstream>
#include <iostream>

#include <glog/logging.h>

#include "MemManager.h"
#include "OCRNetEngine.h"

namespace nvocdr {

bool OCRNetEngine::customInit() {
  // Init Dict

  mDict.clear();

  // todo(shuohanc) could we move this to dict file and release with models, cause the dict also change when we add more charactors?
  if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CTC) {
    mDict.emplace_back("CTCBlank");
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_ATTN) {
    mDict.emplace_back("[GO]");
    mDict.emplace_back("[s]");
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
    mDict.emplace_back("[E]");
  } else {
    LOG(INFO) << "[ERROR] Unsupported decode mode";
  }

  loadDict();

  if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
    mDict.emplace_back("[B]");
    mDict.emplace_back("[P]");
  }

  setupInput(OCRNET_INPUT, {}, true);
  setupOutput(OCRNET_OUTPUT_PROB, {}, true);
  setupOutput(OCRNET_OUTPUT_ID, {}, true);

  mOutputCharLength = getOutputDims(OCRNET_OUTPUT_ID).d[1];
  LOG(INFO) << "decode length: " << mOutputCharLength;

  std::stringstream ss;
  for (auto const& c : mDict) {
    ss << c << ", ";
  }
  LOG(INFO) << "dict: " << ss.str();
  return true;
}
void OCRNetEngine::loadDict() {
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
void OCRNetEngine::decode(Text* const text, size_t idx) {
  if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CTC) {
    decodeCTC(text, idx);
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_ATTN) {
    decodeATTNOrCLIP(text, idx, "[s]");
  } else if (mParam.type == nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP) {
    decodeATTNOrCLIP(text, idx, "[E]");
  }
}

void OCRNetEngine::decodeCTC(Text* const text, size_t idx) {

  int offset = idx * mOutputCharLength;
  float* prob = reinterpret_cast<float*>(
                    mBufManager.getBuffer(getBufName(OCRNET_OUTPUT_PROB), BUFFER_TYPE::HOST)) +
                offset;
  // for trt above 8.6, this node must be casted to int32
  int32_t* cls = reinterpret_cast<int32_t*>(
                     mBufManager.getBuffer(getBufName(OCRNET_OUTPUT_ID), BUFFER_TYPE::HOST)) +
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

void OCRNetEngine::decodeATTNOrCLIP(Text* const text, size_t idx, const std::string& ending) {

  size_t offset = idx * mOutputCharLength;
  float* prob = reinterpret_cast<float*>(
                    mBufManager.getBuffer(getBufName(OCRNET_OUTPUT_PROB), BUFFER_TYPE::HOST)) +
                offset;
  // for trt above 8.6, this node must be casted to int32
  int32_t* cls = reinterpret_cast<int32_t*>(
                     mBufManager.getBuffer(getBufName(OCRNET_OUTPUT_ID), BUFFER_TYPE::HOST)) +
                 offset;

  std::string ret;
  float score = 1.F;
  for (size_t i = 0; i < mOutputCharLength; i++) {
    if (mDict[cls[i]] == ending) {
      break;
    }
    ret += mDict[cls[i]];
    score *= prob[i];
  }
  memcpy(text->text, ret.c_str(), ret.length());
  text->text[ret.length()] = '\0';

  text->text_length = ret.length();
  text->conf = score;
}
}  // namespace nvocdr
