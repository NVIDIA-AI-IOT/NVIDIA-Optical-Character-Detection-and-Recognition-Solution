#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "nvocdr.h"

namespace nvocdr
{
  using namespace nvocdr;

  namespace sample
  {
    class OCDRInferenceSample
    {
    public:
      OCDRInferenceSample() = default;
      OCDRInferenceSample(nvOCDRParam param) : m_param(param)
      {
        m_handler = nvOCDR_initialize(m_param);
      };


      ~OCDRInferenceSample()
      {      }

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
            .num_channel = 3,
            .data = img.data,
            .data_format = HWC};
        nvOCDROutput output;
        nvOCDR_process(m_handler, input, &output);
        return output;
      }

    private:
      nvOCDRParam m_param;
      void *m_handler = nullptr;
      nvOCDRInput m_input;
    };
  }
}