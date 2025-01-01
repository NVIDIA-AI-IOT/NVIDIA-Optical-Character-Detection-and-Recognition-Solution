#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

namespace nvocdr
{
struct ROI
{
    int x;
    int y;
    int width;
    int height;
};

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

struct GRAY_PREPROC_PARAM
{
    float gray_scale;
    float mean;
};


void launch_preprocess_color(uint8_t *src, float* dst, int src_w, int src_h, const cv::Rect& rect, const cv::Size2f& scale, 
    const COLOR_PREPROC_PARAM& param, bool in_bgr, bool out_bgr,
    const cudaStream_t& stream);

void launch_preprocess_gray(uint8_t *src, float* dst, int src_w, int src_h, int dst_w, int dst_h, 
    const GRAY_PREPROC_PARAM& param, bool in_bgr, const cudaStream_t& stream);

} // namespace nvocdr

