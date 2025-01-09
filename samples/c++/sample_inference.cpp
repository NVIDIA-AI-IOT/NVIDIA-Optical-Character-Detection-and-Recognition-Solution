#include "base_sample.hpp"
#include "filesystem"

#include "glog/logging.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    FLAGS_logtostderr = 1;
    FLAGS_colorlogtostderr = 1;

    // instanite a sample 
    nvocdr::sample::OCDRInferenceSample sample;
    // parse the args
    if(!sample.parseArgs(argc, argv)) {
        return 1;
    };

    auto img_path = fs::path(sample.getImagePath());
    LOG(INFO) << "run sample on " << img_path;
    auto dir_path = img_path.parent_path();
    auto img_name = img_path.filename().string();
    auto viz_name = img_name.substr(0, img_name.find('.'));

    // read the image
    cv::Mat origin_image = cv::imread(img_path);
    // do the initalization
    sample.initialize(origin_image.cols, origin_image.rows);
    // get the output;
    const auto output = sample.infer(origin_image);
    // write viz image
    auto const viz = sample.visualize(origin_image, output);
    cv::imwrite(dir_path / (viz_name  + "_result.jpg"), viz);
    
    for(int i = 0;i < 0; ++i) { 
      const auto output = sample.infer(origin_image);
      auto const viz = sample.visualize(origin_image, output);
      cv::imwrite(dir_path / (viz_name  + "_result.jpg"), viz);
    }
    // //     for (size_t i = 0; i < output.num_texts;++i) {
    // //         std::cout<< std::string(output.texts[i].text) << ",";
    // //         for(size_t j = 0; j < 8; ++j) {
    // //             std::cout<< output.texts[i].polygon[j];
    // //             if(j != 7) std::cout<<":";
    // //         }
    // //         std::cout<<"\n";
    // //     }
    
    sample.printStat();

    return 0;
}