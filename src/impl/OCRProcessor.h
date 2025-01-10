#pragma once

#include <vector>
#include <string>

#include "base.h"
#include "tokenizer.h"

namespace nvocdr
{

static constexpr char OCRNET_INPUT[] = "input";
static constexpr char OCRNET_OUTPUT_ID[] = "output_id";
static constexpr char OCRNET_OUTPUT_PROB[] = "output_prob";

static constexpr char CTC_MODEL[] = "OCR_CTC";

static constexpr char CLIP_VISUAL_MODEL[] = "CLIP_VISUAL";
static constexpr char CLIP_VISUAL_INPUT[] = "imgs";
static constexpr char CLIP_VISUAL_OUTPUT_FEATURE[] = "img_feature";
static constexpr char CLIP_VISUAL_OUTPUT_PROBS[] = "visual_decode_probs"; // coarse text, bx26x95, tokenize and put into text_token
static constexpr char CLIP_VISUAL_OUTPUT_ID[] = "tgt_in";

static constexpr char CLIP_TEXT_MODEL[] = "CLIP_TEXT";
static constexpr char CLIP_TEXT_INPUT_TEXT_TOKEN[] = "text_token"; // bx16
static constexpr char CLIP_TEXT_INPUT_FEATURE[] = "img_feature";
static constexpr char CLIP_TEXT_INPUT_ID[] = "tgt_in";
static constexpr char CLIP_TEXT_OUTPUT_LOGITS[] = "logits"; // bx26x95
static constexpr char DICT[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
static constexpr size_t CLIP_VOCAB_SIZE = 32000;

class OCRProcessor : public BaseProcessor<nvOCRParam> {
public:
  bool init() final;
  cv::Size getInputHW() final;
  std::string getInputBufName() final;
  size_t getBatchSize() final;

  OCRProcessor(const nvOCRParam& param);
  void decode(Text* const text, size_t idx);
  bool infer(bool sync_input, const cudaStream_t& stream) final;

 private:
  void initCTC();
  void initATTN();
  void initCLIP();

  void decodeCTC(Text* const text, size_t idx);
//   void decodeATTN(Text* const text, size_t idx, const std::string& ending);
  void decodeCLIP(Text* const text, size_t idx, const std::string& ending);
//   void loadDict();
  void decodeCLIP(const cudaStream_t& stream, const std::string& buf_name);

  size_t mOutputCharLength;
  std::vector<std::string> mDict;
  /// @brief used by CLIP
  std::vector<std::string> mTmpText;
  std::vector<float> mTmpScore;
  size_t mClipCharTypeSize;
  size_t mClipEmbedingSize;
  std::unique_ptr<BPETokenizer> mTokenizer = nullptr;
};
} // namespace nvocdr
