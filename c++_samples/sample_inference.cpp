#include "base_sample.hpp"

int main() {
    nvOCDRParam param;

    param.input_data_format = HWC;

    param.ocd_param.engine_file = (char *)"/home/csh/nvocdr/onnx_models/ocdnet.fp16.engine";
    // param.ocdnet_infer_input_shape[0] = 3;
    // param.ocdnet_infer_input_shape[1] = 736;
    // param.ocdnet_infer_input_shape[2] = 1280;
    param.ocd_param.binarize_threshold = 0.1;
    param.ocd_param.polygon_threshold = 0.3;
    param.ocd_param.max_candidate = 200;
    param.ocd_param.unclip_ratio = 1.5;
    param.ocr_param.engine_file = (char *)"/home/csh/nvocdr/onnx_models/ocrnet.fp16.engine";
    param.ocr_param.dict_file = (char *)"/home/csh/nvocdr/onnx_models/character_list.txt";
    // param.ocrnet_infer_input_shape[0] = 1;
    // param.ocrnet_infer_input_shape[1] = 32;
    // param.ocrnet_infer_input_shape[2] = 100;

    nvocdr::sample::OCDRInferenceSample sample(param);

    std::string img_path = "/home/csh/nvocdr/c++_samples/test_img/doc.jpg";

    cv::Mat origin_image = cv::imread(img_path);

    for (size_t i = 0; i < 10; ++i) {
        sample.infer(origin_image);
    }

    // cv::imwrite("../test.png", viz);

    return 0;
}