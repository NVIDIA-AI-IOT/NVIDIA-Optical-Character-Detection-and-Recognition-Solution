#pragma once

#include <vector>
#include <string>

#include "base.h"

namespace nvocdr
{

constexpr char OCRNET_INPUT[] = "input";
constexpr char OCRNET_OUTPUT_ID[] = "output_id";
constexpr char OCRNET_OUTPUT_PROB[] = "output_prob";

constexpr char CTC_MODEL[] = "OCR_CTC";

constexpr char CLIP_VISUAL_MODEL[] = "CLIP_VISUAL";
constexpr char CLIP_VISUAL_INPUT[] = "imgs";
constexpr char CLIP_VISUAL_OUTPUT_FEATURE[] = "img_feature";
constexpr char CLIP_VISUAL_OUTPUT_PROBS[] = "visual_decode_probs"; // coarse text, bx26x95, tokenize and put into text_token
constexpr char CLIP_VISUAL_OUTPUT_ID[] = "tgt_in";

constexpr char CLIP_TEXT_MODEL[] = "CLIP_TEXT";
constexpr char CLIP_TEXT_INPUT_TEXT_TOKEN[] = "text_token"; // bx16
constexpr char CLIP_TEXT_INPUT_FEATURE[] = "img_feature";
constexpr char CLIP_TEXT_INPUT_ID[] = "tgt_in";
constexpr char CLIP_TEXT_OUTPUT_LOGITS[] = "logits"; // bx26x95
constexpr char CLIP_CHAR[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_{|}~abcdefghijklmnopqrstuvwxyz";

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
  void loadDict();
  void decodeCLIPVisual(const cudaStream_t& stream);

  size_t mOutputCharLength;
  std::vector<std::string> mDict;
  /// @brief used by CLIP
  std::vector<std::vector<char>> mTmpText;
  size_t mClipCharTypeSize;
};
} // namespace nvocdr
