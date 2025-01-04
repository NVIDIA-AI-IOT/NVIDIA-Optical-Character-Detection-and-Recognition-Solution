#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "nvOCDR_impl.h"


namespace py = pybind11;
namespace nvocdr {

class nvOCDRWarp
{
public:
    nvOCDRWarp(const nvOCDParam& ocd_param, const nvOCRParam & ocr_param, const ProcessParam& process_param, 
    const std::string& ocd_model, const std::string& ocr_model, 
    const std::vector<size_t> &input_shape) {
        nvOCDRParam param;
        param.ocd_param = ocd_param;
        param.ocr_param = ocr_param;
        param.process_param = process_param;

        memcpy(param.input_shape, input_shape.data(), 3 * sizeof(size_t));
        strcpy(param.ocd_param.model_file, ocd_model.c_str());
        strcpy(param.ocr_param.model_file, ocr_model.c_str());

        mHandler.reset(new nvOCDR(param));
    };
    // ~nvOCDRWarp();

    std::vector<nvOCDROutput> warpInfer(py::array_t<uchar> &img) {

    };

    // std::vector<std::vector<std::pair<nvocdr::Polygon, std::pair<std::string, float>>>> warpInfer(py::array_t<uchar> &imgs);
    // std::vector<std::vector<std::pair<nvocdr::Polygon, std::pair<std::string, float>>>> warpPatchInfer(py::array_t<uchar> &oriImg, py::array_t<uchar> &imgPatches, const float overlapRate);

private:
    std::unique_ptr<nvOCDR> mHandler;
};
}
// } // namespace nvocdr


