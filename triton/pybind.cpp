
#include "pybind.h"
#include <iomanip>

namespace py = pybind11;
using namespace nvocdr;


nvOCDRWarp::nvOCDRWarp(nvOCDRParam param, std::string& ocd_trt_path, std::string& ocr_trt_path, std::string& ocr_dict_path, std::vector<int> &OCDNetInferShape, std::vector<int> &OCRNetInferShape)
{
    std::cout<< "[nvOCDR] Init..." << std::endl;

    param.ocdnet_trt_engine_path = (char* )ocd_trt_path.c_str();
    param.ocrnet_trt_engine_path = (char* )ocr_trt_path.c_str();
    param.ocrnet_dict_file = (char* )ocr_dict_path.c_str();
    std::cout<< "[nvOCDR] Load OCD tensorRT engine from: " <<param.ocdnet_trt_engine_path<< std::endl;
    std::cout<< "[nvOCDR] Load OCD tensorRT engine from: " <<param.ocrnet_trt_engine_path<< std::endl;

    param.ocdnet_infer_input_shape[0] = OCDNetInferShape[0];
    param.ocdnet_infer_input_shape[1] = OCDNetInferShape[1];
    param.ocdnet_infer_input_shape[2] = OCDNetInferShape[2];
    param.ocrnet_infer_input_shape[0] = OCRNetInferShape[0];
    param.ocrnet_infer_input_shape[1] = OCRNetInferShape[1];
    param.ocrnet_infer_input_shape[2] = OCRNetInferShape[2];

    m_nvOCDRLib = static_cast<nvocdr::nvOCDR*>(nvOCDR_init(param));
}

nvOCDRWarp::~nvOCDRWarp()
{
    std::cout<< "[nvOCDR] DEInit..." <<std::endl;
    nvOCDR_deinit(m_nvOCDRLib);
}


std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>> nvOCDRWarp::warpInfer(py::array_t<uchar> &imgs)
{
    std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>> nvocdr_output;
    if(imgs.ndim() != 4)
    {
        std::cerr<<"[ERROR] The OCDNet inference channel should be 1 or 3."<<std::endl;
        return nvocdr_output;
    }

    nvinfer1::Dims input_shape;
    input_shape.nbDims = imgs.ndim();
    input_shape.d[0] = imgs.shape(0);
    input_shape.d[1] = imgs.shape(1);
    input_shape.d[2] = imgs.shape(2);
    input_shape.d[3] = imgs.shape(3);

    py::buffer_info imgsBuf = imgs.request();
    uchar *ptrImgs = (uchar *)imgsBuf.ptr;
    DeviceBuffer input_buffer( imgs.size(), imgs.itemsize());
    cudaMemcpy(input_buffer.data(), reinterpret_cast<void*>(ptrImgs), imgs.nbytes(), cudaMemcpyHostToDevice);

    {
        AutoProfiler timer("[nvOCDR] batch images infer time: ");
        m_nvOCDRLib->infer(input_buffer.data(), input_shape, nvocdr_output);
    }
    return nvocdr_output;
}


std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>> nvOCDRWarp::warpPatchInfer(py::array_t<uchar> &oriImg, py::array_t<uchar> &imgPatches, const float overlapRate)
{
    std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>> nvocdr_output;
    if(imgPatches.ndim() != 4)
    {
        std::cerr<<"[ERROR] The OCDNet inference channel should be 1 or 3."<<std::endl;
        return nvocdr_output;
    }

    nvinfer1::Dims inputPatcheshape;
    inputPatcheshape.nbDims = imgPatches.ndim();
    inputPatcheshape.d[0] = imgPatches.shape(0);
    inputPatcheshape.d[1] = imgPatches.shape(1);
    inputPatcheshape.d[2] = imgPatches.shape(2);
    inputPatcheshape.d[3] = imgPatches.shape(3);

    nvinfer1::Dims oriImgShape;
    oriImgShape.nbDims = oriImg.ndim();
    oriImgShape.d[0] = oriImg.shape(0);
    oriImgShape.d[1] = oriImg.shape(1);
    oriImgShape.d[2] = oriImg.shape(2);
    oriImgShape.d[3] = oriImg.shape(3);

    py::buffer_info imgsPatchBuf = imgPatches.request();
    py::buffer_info oriImgBuf = oriImg.request();

    uchar *ptrImg = (uchar *)oriImgBuf.ptr;
    uchar *ptrImgPatchs = (uchar *)imgsPatchBuf.ptr;
    // cannot make sure that the high resolution images will have same number of patches , so need to allocate buffer at each infer time 
    DeviceBuffer inputPatchbuffer( imgPatches.size(), imgPatches.itemsize());
    DeviceBuffer oriImgbuffer( oriImg.size(), oriImg.itemsize());

    cudaMemcpy(inputPatchbuffer.data(), reinterpret_cast<void*>(ptrImgPatchs), imgPatches.nbytes(), cudaMemcpyHostToDevice);
    cudaMemcpy(oriImgbuffer.data(), reinterpret_cast<void*>(ptrImg), oriImg.nbytes(), cudaMemcpyHostToDevice);

    {
        AutoProfiler timer("[nvOCDR] High resolution images infer time: ");
        m_nvOCDRLib->inferPatch(oriImgbuffer.data(), inputPatchbuffer.data(), inputPatcheshape, oriImgShape, overlapRate, nvocdr_output);
    }

    return nvocdr_output;
}


PYBIND11_MODULE(nvocdr, m) {
    m.doc() = "nvOCDR lib plugin"; // optional module docstring

    py::class_<nvOCDRWarp>(m, "nvOCDRWarp")
        .def(py::init<nvOCDRParam, std::string&,std::string&,std::string&,std::vector<int>&,std::vector<int>&>())
        .def("warpInfer", &nvOCDRWarp::warpInfer)
        .def("warpPatchInfer", &nvOCDRWarp::warpPatchInfer)
        ;

    py::enum_<DataFormat>(m, "DataFormat")
        .value("NCHW", DataFormat::NCHW)
        .value("NHWC", DataFormat::NHWC)
        .export_values();

    py::enum_<OCRNetDecode>(m, "OCRNetDecode")
        .value("CTC", OCRNetDecode::CTC)
        .value("Attention", OCRNetDecode::Attention)
        .export_values();

    py::class_<nvOCDRParam>(m, "nvOCDRParam")
        .def(py::init<>())
        .def_readwrite("ocdnet_binarize_threshold", &nvOCDRParam::ocdnet_binarize_threshold)
        .def_readwrite("ocdnet_polygon_threshold", &nvOCDRParam::ocdnet_polygon_threshold)
        .def_readwrite("ocdnet_unclip_ratio", &nvOCDRParam::ocdnet_unclip_ratio)
        .def_readwrite("ocdnet_max_candidate", &nvOCDRParam::ocdnet_max_candidate)
        .def_readwrite("upsidedown", &nvOCDRParam::upsidedown)
        .def_readwrite("input_data_format", &nvOCDRParam::input_data_format)
        .def_readwrite("ocrnet_decode", &nvOCDRParam::ocrnet_decode)
        ;

     py::class_<Polygon>(m, "Polygon")
        .def(py::init<>())
        .def_readwrite("x1", &Polygon::x1)
        .def_readwrite("x2", &Polygon::x2)
        .def_readwrite("x3", &Polygon::x3)
        .def_readwrite("x4", &Polygon::x4)
        .def_readwrite("y1", &Polygon::y1)
        .def_readwrite("y2", &Polygon::y2)
        .def_readwrite("y3", &Polygon::y3)
        .def_readwrite("y4", &Polygon::y4)
        ;

}

