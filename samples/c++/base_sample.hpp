#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "nvocdr.h"
#include <boost/program_options.hpp>
namespace nvocdr
{
  namespace po = boost::program_options;

  namespace sample
  {
    class OCDRInferenceSample
    {
    public:
      OCDRInferenceSample() = default;
      void initialize(int width, int height)
      {
        mParam.input_shape[0] = 3;
        mParam.input_shape[1] = height;
        mParam.input_shape[2] = width;
        mHandler = nvOCDR_initialize(mParam);
      };

      ~OCDRInferenceSample()
      {
      }

      cv::Mat visualize(const cv::Mat &img, const nvOCDROutput &texts)
      {
        cv::Mat viz = img.clone();

        for (size_t i = 0; i < texts.num_texts; ++i)
        {
          const auto &text = texts.texts[i];

          std::array<cv::Point, 4> pts;
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
            bool direction = cv::norm(pts[0] - pts[1]) <= cv::norm(pts[1] - pts[2]);

            if (!direction)
            {
              cv::Point2f tmp = pts[3];
              pts[3] = pts[2];
              pts[2] = pts[1];
              pts[1] = pts[0];
              pts[0] = tmp;
            }
            if (mShowScore)
            {
              cv::putText(viz, std::to_string(text.conf), (pts[1] + pts[2]) / 2, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            }
            if (mShowText)
            {
              cv::putText(viz, std::string(text.text), (pts[0] + pts[3]) / 2, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
            }
          }
        }
        return viz;
      }

      nvOCDROutput infer(const cv::Mat &img)
      {
        nvOCDRInput input{
            .data = img.data,
            .data_format = DATAFORMAT_TYPE_HWC};
        nvOCDROutput output;
        nvOCDR_process(mHandler, input, &output);
        return output;
      }

      void printStat() {
        nvOCDR_print_stat(mHandler);
      }

      bool parseArgs(int argc, char **argv)
      {
        po::options_description desc("nvOCDR sample options");
        desc.add_options()
        ("help", "produce help message")
        ("image", po::value<std::string>(&mImagePath), "input image to run test")
        ("strategy", po::value<std::string>()->default_value("smart"), "strategy")
        ("rec_all_direction", po::value<bool>(&mParam.process_param.all_direction)->default_value(false), "use 4 direction to recognize and use best, consume more time")
        ("text_polygon_thresh", po::value<float>(&mParam.process_param.polygon_threshold)->default_value(.3), "threshold for text area polygon")
        ("binary_lower_bound", po::value<float>(&mParam.process_param.binarize_lower_threshold)->default_value(0), "lower bound for binary mask thresholding")
        ("binary_upper_bound", po::value<float>(&mParam.process_param.binarize_upper_threshold)->default_value(0.1), "lower bound for binary mask thresholding")
        ("debug_log", po::value<bool>(&mParam.process_param.debug_log)->default_value(false), "print debug log also")
        ("debug_image", po::value<bool>(&mParam.process_param.debug_image)->default_value(false), "save debug image")
        ("max_candidates", po::value<size_t>(&mParam.process_param.max_candidate)->default_value(100), "maximuam texts to output, if detects exceed this value, lower score ones will be ignored")
        ("ocd_model", po::value<std::string>(), "optical charactor detection model, onnx or engine")
        ("ocr_model", po::value<std::string>(), "optical charactor recognition model, onnx or engine")
        ("ocr_dicts", po::value<std::string>()->default_value("default"), "dictionary for ocr model, txt file. if 'default' was given, use 0-9a-z")
        ("ocr_type", po::value<std::string>()->default_value("CTC"), "the model type you set for ocr_model, choose from CTC/ATTN/CLIP")
        ("ocd_type", po::value<std::string>()->default_value("normal"), "the model type you set for ocd_model, choose from normal/mixnet")
        ("show_text", po::value<bool>(&mShowText)->default_value(true), "viz text")
        ("show_score", po::value<bool>(&mShowScore)->default_value(false), "viz score also")
        ;

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
          std::cout << desc << "\n";
          return false;
        }
        if (!parseStrArgs(vm, "ocd_model", mParam.ocd_param.model_file))
        {
          std::cerr << "parse failed for ocd_model";
          return false;
        }
        if (!parseStrArgs(vm, "ocr_model", mParam.ocr_param.model_file))
        {
          std::cerr << "parse failed for ocr_model";
          return false;
        }
        if (!parseStrArgs(vm, "ocr_dicts", mParam.ocr_param.dict))
        {
          std::cerr << "parse failed for ocr_dicts";
          return false;
        }

        // strategy
        auto strategy = vm["strategy"].as<std::string>();
        if (strategy == "smart") {
          mParam.process_param.strategy = STRATEGY_TYPE_SMART;
        } else if(strategy == "resize") {
          mParam.process_param.strategy = STRATEGY_TYPE_RESIZE_TILE;
        } else if (strategy == "noresize"){
          mParam.process_param.strategy = STRATEGY_TYPE_NORESIZE_TILE;
        } else if (strategy == "resize_full") {
          mParam.process_param.strategy = STRATEGY_TYPE_RESIZE_FULL;
        } else{
          std::cerr << "strategy not supported " <<  strategy <<"\n";
          return false;
        }

        // ocr model type
        auto ocr_type = vm["ocr_type"].as<std::string>();
        if (ocr_type == "CTC") {
          mParam.ocr_param.type = nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CTC;
        } else if(ocr_type == "ATTN") {
          mParam.ocr_param.type = nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_ATTN;
        } else if(ocr_type == "CLIP") {
          mParam.ocr_param.type = nvOCRParam::OCR_MODEL_TYPE::OCR_MODEL_TYPE_CLIP;
        } else{
          std::cerr << "ocr_type not supported " << ocr_type << "\n";
          return false;
        }

        // ocd model type
        auto ocd_type = vm["ocd_type"].as<std::string>();
        if (ocd_type == "normal") {
          mParam.ocd_param.type = nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_NORMAL;
        } else if(ocd_type == "mixnet") {
          mParam.ocd_param.type = nvOCDParam::OCD_MODEL_TYPE::OCD_MODEL_TYPE_MIXNET;
        } else{
          std::cerr << "ocd_type not supported " << ocd_type << "\n";
          return false;
        }


        return true;
      }

      inline std::string getImagePath() const {
        return mImagePath;
      }

    private:
      static bool parseStrArgs(const po::variables_map &vm, const std::string name, char *const dst)
      {
        if (vm.count(name))
        {
          std::string opt = vm[name].as<std::string>();
          if (opt.length() > MAX_FILE_PATH)
          {
            std::cout << "path too long for " << name;
            return false;
          }
          strcpy(dst, opt.c_str());
          return true;
        }
        return false;
      }

      void *mHandler = nullptr;
      nvOCDRParam mParam;
      bool mShowScore;
      bool mShowText;
      std::string mImagePath;
    };
  };
}
