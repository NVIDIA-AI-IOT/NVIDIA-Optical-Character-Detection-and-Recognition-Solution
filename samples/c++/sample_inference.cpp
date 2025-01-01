#include "base_sample.hpp"
// #include "boost/program_options.h"

int main(int argc, char** argv) {
    nvocdr::sample::OCDRInferenceSample sample;
    if(!sample.parseArgs(argc, argv)) {
        return 1;
    };
    
    cv::Mat origin_image = cv::imread(sample.getImagePath());

    sample.initialize(origin_image.cols, origin_image.rows);

    // cv::Mat image;
    // cv::GaussianBlur(origin_image, image, cv::Size(0, 0), 3);
    // cv::addWeighted(origin_image, 2, image, -1, 0, image);
    // cv::resize(image, image, {200,200});
    for(int i = 0;i < 14;++i) {
        const auto output = sample.infer(origin_image);
    //     for (size_t i = 0; i < output.num_texts;++i) {
    //         std::cout<< std::string(output.texts[i].text) << ",";
    //         for(size_t j = 0; j < 8; ++j) {
    //             std::cout<< output.texts[i].polygon[j];
    //             if(j != 7) std::cout<<":";
    //         }
    //         std::cout<<"\n";
    //     }
        auto const viz = sample.visualize(origin_image, output);
        cv::imwrite("viz.png", viz);
    }
    // cv::imwrite("slim_1.png", origin_image(cv::Rect(400, 0, 300, 3904)));
    // cv::imwrite("slim_2.png", origin_image(cv::Rect(0, 300, 3904, 300)));

    sample.printStat();

    return 0;
}