#pragma once

#include <vector>
#include <string>

#include "base.h"

namespace nvocdr
{
constexpr char OCR_PREFIX[] = "OCR";
constexpr char OCRNET_INPUT[] = "input";
constexpr char OCRNET_OUTPUT_ID[] = "output_id";
constexpr char OCRNET_OUTPUT_PROB[] = "output_prob";
constexpr char CTC_MODEL[] = "OCR_CTC";

class OCRProcessor : public BaseProcessor<nvOCRParam> {
public:
  bool init() final;
  // OCRNetEngine() = default;
  cv::Size getInputHW() final;
  std::string getInputBufName() final;
  size_t getBatchSize() final;

  OCRProcessor(const char name[], const nvOCRParam& param);
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
