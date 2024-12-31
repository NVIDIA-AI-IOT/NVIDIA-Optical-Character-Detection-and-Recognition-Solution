#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

namespace nvocdr
{
__host__ __device__ struct ROI
{
    int x;
    int y;
    int width;
    int height;
};

void launch_preprocess(void *src, void* dst, int src_w, int src_h, const cv::Rect& rect, const cv::Size2f& scale, 
    float rbg_scale, float r_mean, float g_mean, float b_mean, float r_std, float g_std, float b_std, bool in_bgr, bool out_bgr,
    const cudaStream_t& stream);

} // namespace nvocdr

