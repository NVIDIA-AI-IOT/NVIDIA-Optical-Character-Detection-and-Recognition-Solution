#include "base_sample.hpp"
// #include "boost/program_options.h"

int main() {
    nvOCDRParam param;

    param.input_data_format = HWC;

    param.process_param.binarize_threshold = 0.1;
    param.process_param.polygon_threshold = 0.3;
    param.process_param.max_candidate = 1000;
    param.process_param.min_pixel_area = 10;

    param.ocd_param.engine_file = (char *)"/home/csh/nvocdr/onnx_models/ocdnet.fp16.engine";

    param.ocr_param.engine_file = (char *)"/home/csh/nvocdr/onnx_models/ocrnet.fp16.engine";
    param.ocr_param.dict_file = (char *)"/home/csh/nvocdr/onnx_models/character_list.txt";


    nvocdr::sample::OCDRInferenceSample sample(param);

    std::string img_path = "/home/csh/nvocdr/c++_samples/test_img/scene_text.jpg";
    std::string img_path2 = "/home/csh/nvocdr/c++_samples/test_img/doc.jpg";
    std::vector<std::string> imgs{img_path2,};


    for (size_t i = 0; i < 1; ++i) {
        cv::Mat origin_image = cv::imread(imgs[i%1]);
        const auto output = sample.infer(origin_image);
        auto const viz = sample.visualize(origin_image, output);
        cv::imwrite("viz.png", viz);
    }

    return 0;
}