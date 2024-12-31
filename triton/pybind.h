#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "nvOCDR.h"

namespace nvocdr
{
    
namespace py = pybind11;


class nvOCDRWarp 
{
public:
    nvOCDRWarp(nvOCDRParam param, std::string& ocd_trt_path, std::string& ocr_trt_path, std::string& ocr_dict_path,std::vector<int> &OCDNetInferShape, std::vector<int> &OCRNetInferShape);
    ~nvOCDRWarp();

    std::vector<std::vector<std::pair<nvocdr::Polygon, std::pair<std::string, float>>>> warpInfer(py::array_t<uchar> &imgs);
    std::vector<std::vector<std::pair<nvocdr::Polygon, std::pair<std::string, float>>>> warpPatchInfer(py::array_t<uchar> &oriImg, py::array_t<uchar> &imgPatches, const float overlapRate);

private:
    std::unique_ptr<nvOCDR> mHandler;
};
} // namespace nvocdr



