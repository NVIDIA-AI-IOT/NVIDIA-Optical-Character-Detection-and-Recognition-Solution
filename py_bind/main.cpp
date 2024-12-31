#include <pybind11/pybind11.h>
#include "py_nvocdr.hpp"

namespace py = pybind11;

using namespace nvocdr;
PYBIND11_MODULE(nvocdr_python, m) {
    m.doc() = "nvocdr python binding";

    
    py::class_<nvOCDRWarp>(m, "nvOCDRWarp")
        .def(py::init<const nvOCDParam&, const nvOCRParam &, const ProcessParam&, const std::string&,const std::string&,const std::vector<size_t> &>())
        .def("warpInfer", &nvOCDRWarp::warpInfer)
        ;

    py::enum_<nvOCDParam::OCD_MODEL_TYPE>(m, "OCD_MODEL_TYPE")
        .value("OCD_MODEL_TYPE_NORMAL", nvOCDParam::OCD_MODEL_TYPE_NORMAL)
        .value("OCD_MODEL_TYPE_MIXNET", nvOCDParam::OCD_MODEL_TYPE_MIXNET)
        .export_values();
    
    py::enum_<nvOCRParam::OCR_MODEL_TYPE>(m, "OCR_MODEL_TYPE")
        .value("OCR_MODEL_TYPE_CTC", nvOCRParam::OCR_MODEL_TYPE_CTC)
        .value("OCR_MODEL_TYPE_ATTN", nvOCRParam::OCR_MODEL_TYPE_ATTN)
        .value("OCR_MODEL_TYPE_CLIP", nvOCRParam::OCR_MODEL_TYPE_CLIP)
        .export_values();

    py::class_<nvOCDParam>(m, "nvOCDParam")
        .def(py::init<>())
        .def_readwrite("type", &nvOCDParam::type)
        // .def_readwrite("model_file", &nvOCDParam::model_file)
        .def("model_file", [](const std::string& m) {
            return m.c_str();
        })
        ;

    py::class_<nvOCRParam>(m, "nvOCRParam")
        .def(py::init<>())
        .def_readwrite("type", &nvOCRParam::type);

    py::class_<ProcessParam>(m, "ProcessParam")
        .def(py::init<>())
        .def_readwrite("max_candidate", &ProcessParam::max_candidate);


    //     py::class_<nvOCDRParam>(m, "nvOCDRParam")
    //     .def(py::init<>())
    //     .def_readwrite("ocdnet_binarize_threshold", &nvOCDRParam::ocdnet_binarize_threshold)
    //     .def_readwrite("ocdnet_polygon_threshold", &nvOCDRParam::ocdnet_polygon_threshold)
    //     .def_readwrite("ocdnet_unclip_ratio", &nvOCDRParam::ocdnet_unclip_ratio)
    //     .def_readwrite("ocdnet_max_candidate", &nvOCDRParam::ocdnet_max_candidate)
    //     .def_readwrite("upsidedown", &nvOCDRParam::upsidedown)
    //     .def_readwrite("input_data_format", &nvOCDRParam::input_data_format)
    //     .def_readwrite("ocrnet_decode", &nvOCDRParam::ocrnet_decode)
}