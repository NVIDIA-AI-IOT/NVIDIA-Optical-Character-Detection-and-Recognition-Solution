#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "nvocdr.h"

namespace nvocdr {
    using namespace nvocdr;

namespace sample {
    class OCDRInferenceSample {
    public:
      OCDRInferenceSample() = default;
      OCDRInferenceSample(nvOCDRParam param) : m_param(param) {
            // nvOCDRInput input;
            // m_input.device_type = GPU;
            // input.shape[0] = 1;
            // input.shape[3] = 3;
            m_handler = nvOCDR_initialize(m_param);
      };
      


      // void init() {
      //   if (!m_handler) {
              
      //         std::cout<<  "init success\n";
      //   } else {
      //     std::cout<<  "skip init\n";

      //   }
      // }

      ~OCDRInferenceSample() {
        // nvOCDR_deinit(m_handler);
      }

      cv::Mat visualize(cv::Mat img, nvOCDROutput texts,
              int input_width, int input_height)
        {
            // float scale_x = static_cast<float>(img.size().width) / static_cast<float>(input_width);
            // float scale_y = static_cast<float>(img.size().height) / static_cast<float>(input_height);
            // for(int i = 0; i < texts.text_cnt[0]; ++i)
            // {
            //     std::string output_text(texts.text_ptr[i].ch);
            //     std::stringstream output_conf;
            //     output_conf << std::fixed << std::setprecision(3) << texts.text_ptr[i].conf;
            //     float x1 = texts.text_ptr[i].polys[0];
            //     float y1 = texts.text_ptr[i].polys[1];
            //     float x2 = texts.text_ptr[i].polys[2];
            //     float y2 = texts.text_ptr[i].polys[3];
            //     float x3 = texts.text_ptr[i].polys[4];
            //     float y3 = texts.text_ptr[i].polys[5];
            //     float x4 = texts.text_ptr[i].polys[6];
            //     float y4 = texts.text_ptr[i].polys[7];
            //     cv::line(img, cv::Point((int)(x1 * scale_x), (int)(y1 * scale_y)), cv::Point((int)(x2 * scale_x), (int)(y2 * scale_y)), cv::Scalar(0, 255, 0), 1);
            //     cv::line(img, cv::Point((int)(x2 * scale_x), (int)(y2 * scale_y)), cv::Point((int)(x3 * scale_x), (int)(y3 * scale_y)), cv::Scalar(0, 255, 0), 1);
            //     cv::line(img, cv::Point((int)(x3 * scale_x), (int)(y3 * scale_y)), cv::Point((int)(x4 * scale_x), (int)(y4 * scale_y)), cv::Scalar(0, 255, 0), 1);
            //     cv::line(img, cv::Point((int)(x4 * scale_x), (int)(y4 * scale_y)), cv::Point((int)(x1 * scale_x), (int)(y1 * scale_y)), cv::Scalar(0, 255, 0), 1);
            //     cv::putText(img, output_text, cv::Point(x4 * scale_x, y4 * scale_y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
            // }
            return img;
        }
        
        void infer(const cv::Mat &img) {

            // std::vector<uint8_t> buf(img.total() * img.elemSize());

            // cv::Mat 
            
            nvOCDRInput input {
                  .height = static_cast<size_t>(img.rows),
                  .width = static_cast<size_t>(img.cols),
                  .num_channel = 3,
                  .data = img.data,
                  .data_format = HWC
            };
            nvOCDROutput output;
            nvOCDR_process(m_handler, input, &output);

        }

        // cv::Mat infer_visualize(const cv::Mat &img) {
        //       nvOCDRInput input;
        //       input.device_type = GPU;
        //       input.shape[0] = 1;
        //       input.shape[1] = img.size().height;
        //       input.shape[2] = img.size().width;
        //       input.shape[3] = 3;
        //       size_t item_size = input.shape[1] * input.shape[2] * input.shape[3] * sizeof(uchar);
        //       std::cout<< "input buffer size: " << item_size << "\n";
        //       cudaMalloc(&input.mem_ptr, item_size);
        //       cudaMemcpy(input.mem_ptr, reinterpret_cast<void*>(img.data), item_size, cudaMemcpyHostToDevice);

        //       // Do inference
        //       nvOCDROutputMeta output;
        //       // simple inference
        //       nvOCDR_inference(input, &output, m_handler);
        //       std::cout<< "finish infer\n";

        //       // Visualize the output
        //       int offset = 0;
        //       cv::Mat viz = visualize(img, output, input.shape[2], input.shape[1]);

        //       // Destroy the resoures
        //       free(output.text_ptr);
        //       cudaFree(input.mem_ptr);
        //       // 
        //       return cv::Mat();
        // }


    private:
      nvOCDRParam m_param;
      void* m_handler = nullptr;
      nvOCDRInput m_input;
    };
}
}