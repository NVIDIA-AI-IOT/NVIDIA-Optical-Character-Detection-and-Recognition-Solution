#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "nvocdr.h"
#include <boost/program_options.hpp>
namespace nvocdr
{
namespace po = boost::program_options;

  namespace sample
  {
    static nvOCDRParam DEFAULT_PARAM {
      .ocd_param = {
        .model_file = '\0',
        .batch_size = 0
      },
      .ocr_param = {
        .model_file = '\0',
        .dict = '\0',
        .batch_size = 0,
        .mode = nvOCRParam::OCR_DECODE_TYPE::DECODE_TYPE_CTC
      },
      .process_param {
        .strategy = STRATEGY_TYPE_SMART,
        .max_candidate = 200,
        .polygon_threshold = 0.3F,
        .binarize_threshold = 0.1F,
        .min_pixel_area = 10
      }
    };
    class OCDRInferenceSample
    {
    public:
      OCDRInferenceSample() = default;
      void initialize() 
      {
        mHandler = nvOCDR_initialize(mParam);
      };


      ~OCDRInferenceSample()
      {   }

      cv::Mat visualize(const cv::Mat &img, const nvOCDROutput &texts, bool show_score = false,
      bool show_text=true)
      {
        cv::Mat viz = img.clone();

        for (size_t i = 0; i < texts.num_texts; ++i)
        {
          const auto &text = texts.texts[i];
          std::vector<cv::Point> pts(4);
          for (size_t j = 0; j < 4; ++j)
          {
            cv::Point pt1(text.polygon[j * 2], text.polygon[j * 2 + 1]);
            size_t next = (j + 1) % 4;
            cv::Point pt2(text.polygon[next * 2], text.polygon[next * 2 + 1]);
            cv::line(viz, pt1, pt2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            pts[j] = pt1;
          }

          if (text.text_length > 0)
          {
            if (show_score) {
            cv::putText(viz, std::to_string(text.conf), pts[0], cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            }
            if (show_text) {
            cv::putText(viz, std::string(text.text), pts[2], cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
            }
          }
        }
        return viz;
      }

      nvOCDROutput infer(const cv::Mat &img)
      {
        nvOCDRInput input{
            .height = static_cast<size_t>(img.rows),
            .width = static_cast<size_t>(img.cols),
            .num_channel = img.channels(),
            .data = img.data,
            .data_format = DATAFORMAT_TYPE_HWC};
        nvOCDROutput output;
        nvOCDR_process(mHandler, input, &output);
        return output;
      }

      bool parse_args(int argc, char** argv) {
          po::options_description desc("nvOCDR options");
          desc.add_options()
               ("help", "produce help message")
               ("max_candidates", po::value<size_t>(&mParam.process_param.max_candidate)->default_value(100), "maximuam texts to output, if detects exceed this value, lower score ones will be ignored")
               ("ocd_model", po::value<std::string>(), "optical charactor detection model, onnx or engine")
               ("ocr_model", po::value<std::string>(), "optical charactor recognition model, onnx or engine")
               ("ocr_dicts", po::value<std::string>()->default_value("default"), "dictionary for ocr model, txt file. if 'default' was given, use 0-9a-z")
               ;

          po::variables_map vm;
          po::store(po::parse_command_line(argc, argv, desc), vm);
          po::notify(vm);    

          if (vm.count("help")) {
              std::cout << desc << "\n";
              return false;
          }
          if (vm.count("ocd_model") && vm.count("ocr_model")) {
            std::string ocd_model = vm["ocd_model"].as<std::string>();
            std::string ocr_model = vm["ocr_model"].as<std::string>();

            if (ocd_model.length() > MAX_FILE_PATH || ocr_model.length() > MAX_FILE_PATH) {
              std::cout<< "path too long";
            }
            strcpy(mParam.ocd_param.model_file, ocd_model.c_str());
            strcpy(mParam.ocr_param.model_file, ocr_model.c_str());
          }
          std::string ocr_dicts = vm["ocr_dicts"].as<std::string>();

          // ocr_dicts
          if (ocr_dicts.length() > MAX_FILE_PATH) {
              std::cout<< "path too long";
          }
          strcpy(mParam.ocr_param.dict, ocr_dicts.c_str());

          // max_candidates
          // mParam.process_param.max_candidate = 

          
          return true;
      }

    private:
      void *mHandler = nullptr;
      nvOCDRParam mParam = DEFAULT_PARAM;
    };
  };
}

