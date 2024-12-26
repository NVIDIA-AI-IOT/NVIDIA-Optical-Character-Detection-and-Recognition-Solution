#include "base_sample.hpp"
// #include "boost/program_options.h"

int main(int argc, char** argv) {
    nvocdr::sample::OCDRInferenceSample sample;
    if(!sample.parse_args(argc, argv)) {
        return 1;
    };
    sample.initialize();

    cv::Mat origin_image = cv::imread(sample.mImagePath);
    // cv::Mat image;
    // cv::GaussianBlur(origin_image, image, cv::Size(0, 0), 3);
    // cv::addWeighted(origin_image, 2, image, -1, 0, image);
    // cv::resize(image, image, {200,200});
    // for(int i = 0;i < 10;++i)
        const auto output = sample.infer(origin_image);
    auto const viz = sample.visualize(origin_image, output);
    cv::imwrite("viz.png", viz);
    // cv::imwrite("slim_1.png", origin_image(cv::Rect(400, 0, 300, 3904)));
    // cv::imwrite("slim_2.png", origin_image(cv::Rect(0, 300, 3904, 300)));
    


    return 0;
}