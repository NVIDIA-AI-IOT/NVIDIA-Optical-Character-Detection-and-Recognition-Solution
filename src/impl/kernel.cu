#include <iostream>
#include <cuda_runtime.h>

#include "kernel.h"
#include "macro.h"

inline int divUp(int a, int b)
{
    assert(b > 0);
    return ceil((float)a / b);
};


template<typename T>
__host__ __device__ __forceinline__ T bordConstant(T* src, int y, int x, int c, int h, int w ,  int bordVal = 0)
{
    if((float)x >= 0 && x < w && (float)y >= 0 && y < h)
    {   
        // printf("[%d, %d, %d, %d]", x, y, c, *(src + (y * w  + x) * 3 + c));
        return *(src + (y * w  + x) * 3 + c);
    }
    else 
    {
        return (T)bordVal;
    }
}

template<typename srcDtype, typename dstDtype>
__host__ __device__ __forceinline__ dstDtype linearInterp(srcDtype* src, float y, float x, int c, int h, int w)
{
    const int x1 = int(x);
    const int y1 = int(y);
    const int x2 = x1 + 1;
    const int y2 = y1 + 1;
    float out = 0;

    srcDtype src_reg = bordConstant(src, y1, x1, c, h, w);
    out               = out + src_reg * ((x2 - x) * (y2 - y));

    src_reg = bordConstant(src, y1, x2, c, h, w);
    out     = out + src_reg * ((x - x1) * (y2 - y));

    src_reg = bordConstant(src, y2, x1, c, h, w);
    out     = out + src_reg * ((x2 - x) * (y - y1));

    src_reg = bordConstant(src, y2, x2, c, h, w);
    out     = out + src_reg * ((x - x1) * (y - y1));

    return (dstDtype)out;
}


//8UC3, origin size -> 32FC3, normalized
// grid = [dst_h / 32, dst_w / 32], block = [32, 32]
template<bool IN_BGR, bool OUT_BGR>
__global__ void fused_preprocess_kernel(uint8_t* src, float* dst, int src_w, int src_h, float scale_x, float scale_y,
       nvocdr::ROI roi, // roi on dst image
       float rbg_scale, float r_mean, float g_mean, float b_mean, float r_std, float g_std, float b_std) {
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;


    size_t elem_size = sizeof(float);
    size_t channel_size = roi.width * roi.height;
    if (dst_x < roi.width && dst_y < roi.height) {
        float fy = (float)((dst_y + roi.y + 0.5f) * scale_y - 0.5f) ;
        float fx = (float)((dst_x + roi.x + 0.5f) * scale_x - 0.5f) ;

        float b = linearInterp<uint8_t, float>(src, fy, fx, 0, src_h, src_w);
        float g = linearInterp<uint8_t, float>(src, fy, fx, 1, src_h, src_w);
        float r = linearInterp<uint8_t, float>(src, fy, fx, 2, src_h, src_w);
        // if (dst_y == 1 && dst_x == 0) {
        //     printf("[%.1f %.1f| %d %d %d | %.1f %.1f %.1f]\n", fx, fy, src[0], src[1], src[2], b, g, r);
        // }

        // (0, 0) = [77, 76, 66] correct
        // (1, 0) = [84, 83, 73]
        // (1, 1) = [84, 83, 73]
        // (0, 1) = [78, 77, 67]
        // if (dst_y == 1 && dst_x == 1) {
        //     printf("[%d %d %d]", (dst_y * src_w  + dst_x) * 3, (dst_y * src_w  + dst_x) * 3 + 1, (dst_y * src_w  + dst_x) * 3+2);
        //     printf("[%.1f %.1f| %d %d %d | %.1f %.1f %.1f]\n", fx, fy, src[(dst_y * src_w  + dst_x) * 3], src[(dst_y * src_w  + dst_x) * 3 + 1], src[(dst_y * src_w  + dst_x) * 3 + 2], b, g, r);
        // }

        // float g = 0; float r = 0;

        if (!IN_BGR) { // input = rgb, which mean need to swap b <-> r
          float tmp = b;
          b = r;
          r = tmp;
        }

        b = (b / rbg_scale - b_mean) / b_std;
        g = (b / rbg_scale - g_mean) / g_std;
        r = (r / rbg_scale - r_mean) / r_std;
        // printf("[ %.3f,%.3f, %.3f]", b, g, r);

        // if (0 * channel_size + dst_y * roi.width + dst_x >= roi.width * roi.height * 3) {
        //     printf("[%d, %d,]", dst_x, dst_y);
        // }

        // printf("[%.1f,%.1f,%.1f,]", r, g, b);

        if (OUT_BGR) {
            *(dst + 0 * channel_size + dst_y * roi.width + dst_x) = b;
            *(dst + 1 * channel_size + dst_y * roi.width + dst_x) = g;
            *(dst + 2 * channel_size + dst_y * roi.width + dst_x ) = r;
        } else {
            *(dst + 0 * channel_size + dst_y * roi.width + dst_x) = r;
            *(dst + 1 * channel_size + dst_y * roi.width + dst_x) = g;
            *(dst + 2 * channel_size + dst_y * roi.width + dst_x) = b;
        }
    }
}

namespace nvocdr
{
static constexpr size_t BLOCK_SIZE_X = 32;
static constexpr size_t BLOCK_SIZE_Y = 32;

void launch_preprocess(void *src, void* dst, int src_w, int src_h, const cv::Rect& rect, const cv::Size2f& scale, 
    float rbg_scale, float r_mean, float g_mean, float b_mean, float r_std, float g_std, float b_std, bool in_bgr, bool out_bgr,
    const cudaStream_t& stream) {
    // for(size_t i = 0; i < rois.size(); ++i) {
        // const auto &rect = rois[i];
        nvocdr::ROI roi;
        roi.x = rect.tl().x;
        roi.y = rect.tl().y;
        roi.width = rect.size().width;
        roi.height = rect.size().height;

        dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y); // 1024
        dim3 grid(divUp(roi.width, BLOCK_SIZE_X), divUp(roi.height, BLOCK_SIZE_Y));        
        // dim3 grid(1, 1);        
        // std::cout<< "launch grid: "<< grid.x << " " << grid.y << "\n";
        // std::cout<< "src: "<< src_w << " " << src_h << "\n";
        // std::cout<< "scale: "<< scale.width << " " << scale.height << "\n";
        // std::cout<< roi.x << " " << roi.y << " " << roi.width << " " << roi.height << "\n\n"; 
        if (in_bgr && out_bgr) {
            fused_preprocess_kernel<true, true><<<grid, block, 0, stream>>>(
            static_cast<uint8_t*>(src), static_cast<float*>(dst), 
            src_w, src_h, scale.width, scale.height, roi,
        rbg_scale, r_mean, g_mean, b_mean, r_std, g_std, b_std);
        } else if (!in_bgr && out_bgr) {
            fused_preprocess_kernel<false, true><<<grid, block, 0, stream>>>(
            static_cast<uint8_t*>(src), static_cast<float*>(dst), 
            src_w, src_h, scale.width, scale.height, roi,
            rbg_scale, r_mean, g_mean, b_mean, r_std, g_std, b_std);
        } else if(in_bgr && !out_bgr) {
            fused_preprocess_kernel<true, false><<<grid, block, 0, stream>>>(
            static_cast<uint8_t*>(src), static_cast<float*>(dst), 
            src_w, src_h, scale.width, scale.height, roi,
            rbg_scale, r_mean, g_mean, b_mean, r_std, g_std, b_std);           
        } else {
            fused_preprocess_kernel<false, false><<<grid, block, 0, stream>>>(
            static_cast<uint8_t*>(src), static_cast<float*>(dst), 
            src_w, src_h, scale.width, scale.height, roi,
            rbg_scale, r_mean, g_mean, b_mean, r_std, g_std, b_std);    
        }
        
    // }
    // cudaStreamSynchronize(stream);
    // checkKernelErrors();

}
} // namespace nvocdr




