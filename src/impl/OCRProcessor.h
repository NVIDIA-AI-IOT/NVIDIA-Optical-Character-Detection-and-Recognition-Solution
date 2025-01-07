#pragma once

#include <vector>
#include <string>

#include "base.h"

namespace nvocdr
{
// todo standarize these 
// constexpr char OCR_PREFIX[] = "OCR"; // prefix for 

constexpr char OCRNET_INPUT[] = "input";
constexpr char OCRNET_OUTPUT_ID[] = "output_id";
constexpr char OCRNET_OUTPUT_PROB[] = "output_prob";

constexpr char CTC_MODEL[] = "OCR_CTC";

constexpr char CLIP_VISUAL_MODEL[] = "CLIP_VISUAL";
constexpr char CLIP_VISUAL_INPUT[] = "imgs";
constexpr char CLIP_VISUAL_OUTPUT_FEATURE[] = "img_feature";
constexpr char CLIP_VISUAL_OUTPUT_PROBS[] = "visual_decode_probs"; // bx26x95, tokenize and put o text_token
constexpr char CLIP_VISUAL_OUTPUT_ID[] = "tgt_in";

constexpr char CLIP_TEXT_MODEL[] = "CLIP_TEXT";
constexpr char CLIP_TEXT_INPUT_TEXT_TOKEN[] = "text_token"; // bx16
constexpr char CLIP_TEXT_INPUT_FEATURE[] = "img_feature";
constexpr char CLIP_TEXT_INPUT_ID[] = "tgt_in";
constexpr char CLIP_TEXT_OUTPUT_LOGITS[] = "logits";



class OCRProcessor : public BaseProcessor<nvOCRParam> {
public:
  bool init() final;
  // OCRNetEngine() = default;
  cv::Size getInputHW() final;
  std::string getInputBufName() final;
  size_t getBatchSize() final;

  OCRProcessor(const nvOCRParam& param);
  void decode(Text* const text, size_t idx);

 private:

  void initCTC();
  void initATTN();
  void initCLIP();

  void decodeCTC(Text* const text, size_t idx);
  void decodeATTNOrCLIP(Text* const text, size_t idx, const std::string& ending);
  void loadDict();
  // void decodeCLIP(Text * const text, size_t idx);
  size_t mOutputCharLength;
  std::vector<std::string> mDict;
};
} // namespace nvocdr
