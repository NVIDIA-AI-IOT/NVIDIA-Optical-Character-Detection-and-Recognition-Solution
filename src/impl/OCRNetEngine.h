#pragma once
#include <string>
#include <vector>
#include "MemManager.h"
#include "TRTEngine.h"
#include "nvocdr.h"


namespace nvocdr
{
    

constexpr char OCR_PREFIX[] = "OCR";
constexpr char OCRNET_INPUT[] = "input";
constexpr char OCRNET_OUTPUT_ID[] = "output_id";
constexpr char OCRNET_OUTPUT_PROB[] = "output_prob";

class OCRNetEngine: public OCRTRTEngine
{
    public:
        bool customInit() final;
        // OCRNetEngine() = default;
        OCRNetEngine(const char name[], const nvOCRParam& param) : OCRTRTEngine(name, param) { };
        void decode(Text * const text, size_t idx);

    private:
        void decodeCTC( Text * const text, size_t idx);
        size_t mOutputCharLength;
        std::vector<std::string> mDict;
};
}