#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

namespace nvocdr
{
struct COLOR_PREPROC_PARAM
{
    float rgb_scale;
    float r_mean;
    float g_mean;
    float b_mean;
    float r_std;
    float g_std;
    float b_std;
};

// void launch_preprocess_color(uint8_t *src, float* dst, int src_w, int src_h, const cv::Rect& rect, const cv::Size2f& scale, 
//     const COLOR_PREPROC_PARAM& param, bool in_bgr, bool out_bgr,
//     const cudaStream_t& stream);

// void launch_preprocess_gray(uint8_t *src, float* dst, int src_w, int src_h, int dst_w, int dst_h, 
//     const GRAY_PREPROC_PARAM& param, bool in_bgr, const cudaStream_t& stream);

// void launch_warp_perspective_rgb(float *src, float* dst, int dst_w, int dst_h, const cv::Rect& rect, const cudaStream_t& stream);

// run convert gray + preprocess + warp perspective  fused 
// void launch_fused_preprocess_warp_perspective_gray(uint8_t *src, float* dst, int src_w, int src_h, const cv::Rect& rect, const COLOR_PREPROC_PARAM& param, const cudaStream_t& stream,
//                           double m1,double m2,double m3,double m4,double m5,double m6,double m7,double m8,double m9);

template<bool IN_BGR, bool OUT_BGR, bool OUT_GRAY>
void launch_fused_warp_perspective(uint8_t *src, float* dst, const cv::Size src_size, const cv::Size& dst_size, 
                                   const COLOR_PREPROC_PARAM& param, const cudaStream_t& stream, const cv::Mat& matrix);

} // namespace nvocdr

