#pragma once

#include <string>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "MemManager.h"
#include "TRTEngine.h"
#include "kernel.h"
#include "nvocdr.h"

// #define OCD_DEBUG

namespace nvocdr
{
constexpr char OCD_PREFIX[] = "OCD";
constexpr char OCDNET_INPUT[] = "input";
constexpr char OCDNET_OUTPUT[] = "pred";

class OCDNetEngine : public OCDTRTEngine
{
    public:
        bool customInit() final;
        OCDNetEngine(const char name[], const nvOCDParam& param) : OCDTRTEngine(name, param) { };
    private:
};
}
