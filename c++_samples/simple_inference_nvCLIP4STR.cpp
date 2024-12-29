#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "nvocdr.h"


int visualize(std::string img_path, nvOCDROutputMeta texts,
              int input_width, int input_height)
{
    cv::Mat img = cv::imread(img_path);
    float scale_x = static_cast<float>(img.size().width) / static_cast<float>(input_width);
    float scale_y = static_cast<float>(img.size().height) / static_cast<float>(input_height);
    for(int i = 0; i < texts.text_cnt[0]; ++i)
    {
        std::string output_text(texts.text_ptr[i].ch);
        std::stringstream output_conf;
        output_conf << std::fixed << std::setprecision(3) << texts.text_ptr[i].conf;
        float x1 = texts.text_ptr[i].polys[0];
        float y1 = texts.text_ptr[i].polys[1];
        float x2 = texts.text_ptr[i].polys[2];
        float y2 = texts.text_ptr[i].polys[3];
        float x3 = texts.text_ptr[i].polys[4];
        float y3 = texts.text_ptr[i].polys[5];
        float x4 = texts.text_ptr[i].polys[6];
        float y4 = texts.text_ptr[i].polys[7];
        cv::line(img, cv::Point((int)(x1 * scale_x), (int)(y1 * scale_y)), cv::Point((int)(x2 * scale_x), (int)(y2 * scale_y)), cv::Scalar(0, 255, 0), 1);
        cv::line(img, cv::Point((int)(x2 * scale_x), (int)(y2 * scale_y)), cv::Point((int)(x3 * scale_x), (int)(y3 * scale_y)), cv::Scalar(0, 255, 0), 1);
        cv::line(img, cv::Point((int)(x3 * scale_x), (int)(y3 * scale_y)), cv::Point((int)(x4 * scale_x), (int)(y4 * scale_y)), cv::Scalar(0, 255, 0), 1);
        cv::line(img, cv::Point((int)(x4 * scale_x), (int)(y4 * scale_y)), cv::Point((int)(x1 * scale_x), (int)(y1 * scale_y)), cv::Scalar(0, 255, 0), 1);
        cv::putText(img, output_text, cv::Point(x4 * scale_x, y4 * scale_y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
    }
    cv::imwrite(img_path+"_v.jpg", img);

    return 0;
}

void textFilter(nvOCDROutputMeta texts, const std::string keeped_charset, bool lowercase_only=true, bool uppercase_only=false)
{
    // we only have 1 image in this sample
     for(int i = 0; i < texts.text_cnt[0]; ++i)
    {
        std::string filted_text = "";
        std::string output_text(texts.text_ptr[i].ch);
        if(lowercase_only)
        {
            std::transform(output_text.begin(), output_text.end(), output_text.begin(), ::tolower);
        }
        else if(uppercase_only)
        {
            std::transform(output_text.begin(), output_text.end(), output_text.begin(), ::toupper);
        }
        for(int idx=0; idx<output_text.size(); idx++)
        {
            if (keeped_charset.find(output_text[idx]) == std::string::npos)
            {
        
                continue;
            }
            filted_text += output_text[idx];
        }
        strncpy(texts.text_ptr[i].ch, filted_text.c_str(), (size_t) MAX_CHARACTER_LEN - 1);

    }

}


int main()
{

    // Init the nvOCDR lib
    // Please pay attention to the following parameters. You may need to change them according to different models.
    nvOCDRParam param;
    param.input_data_format = NHWC;
    param.ocdnet_trt_engine_path = (char *)"/home/binz/ssd_4t/NVIDIA-Optical-Character-Detection-and-Recognition-Solution/onnx_models/ocdnet_vit.fp16.engine";
    param.ocdnet_infer_input_shape[0] = 3;
    param.ocdnet_infer_input_shape[1] = 736;
    param.ocdnet_infer_input_shape[2] = 1280;
    param.ocdnet_binarize_threshold = 0.1;
    param.ocdnet_polygon_threshold = 0.3;
    param.ocdnet_max_candidate = 200;
    param.ocdnet_unclip_ratio = 1.5;
    param.ocrnet_trt_engine_path = (char *)"/home/binz/CLIP4STR_nvCLIP/trained_with_nvclip/best_ckpt/vl4str_2024-11-19-06-48-47_checkpoints_epoch_9-step_15580-val_accuracy_71.1684-val_NED_79.9133.visual.sim.fp16.engine";
    param.ocrnet_text_trt_engine_path = (char *)"/home/binz/CLIP4STR_nvCLIP/trained_with_nvclip/best_ckpt/vl4str_2024-11-19-06-48-47_checkpoints_epoch_9-step_15580-val_accuracy_71.1684-val_NED_79.9133.text.sim.fp16.engine";
    param.ocrnet_vocab_file = (char *)"/home/binz/CLIP4STR_nvCLIP/code/CLIP4STR/strhub/clip/bpe_simple_vocab_16e6.txt";
    param.ocrnet_vocab_size = 32000;
    param.ocrnet_dict_file = (char *)"/home/binz/ssd_4t/NVIDIA-Optical-Character-Detection-and-Recognition-Solution/onnx_models/character_list_clip4str";
    param.ocrnet_infer_input_shape[0] = 3;
    param.ocrnet_infer_input_shape[1] = 224;
    param.ocrnet_infer_input_shape[2] = 224;
    param.ocrnet_decode = Transformer;
    param.ocrnet_only_alnum = false;
    param.ocrnet_only_lowercase = false;

    nvOCDRp nvocdr_ptr = nvOCDR_init(param);

    // Load the input
    const char* img_path = "/home/binz/ssd_4t/NVIDIA-Optical-Character-Detection-and-Recognition-Solution/c++_samples/test_img/nvocdr.jpg";
    cv::Mat img = cv::imread(img_path);
    nvOCDRInput input;
    input.device_type = GPU;
    input.shape[0] = 1;
    input.shape[1] = img.size().height;
    input.shape[2] = img.size().width;
    input.shape[3] = 3;
    size_t item_size = input.shape[1] * input.shape[2] * input.shape[3] * sizeof(uchar);
    cudaMalloc(&input.mem_ptr, item_size);
    cudaMemcpy(input.mem_ptr, reinterpret_cast<void*>(img.data), item_size, cudaMemcpyHostToDevice);

    // Do inference
    nvOCDROutputMeta output;
    // simple inference
    nvOCDR_inference(input, &output, nvocdr_ptr);
    
    // filter the output text, and covert to lowercase
    std::string keeped_charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";;
    textFilter(output, keeped_charset, param.ocrnet_only_lowercase);

    // Visualize the output
    int offset = 0;
    visualize(img_path, output, input.shape[2], input.shape[1]);
    std::cout << "sample infer done, results save to  " << img_path << std::endl;
    // Destroy the resoures
    free(output.text_ptr);
    cudaFree(input.mem_ptr);
    nvOCDR_deinit(nvocdr_ptr);

    return 0;
}