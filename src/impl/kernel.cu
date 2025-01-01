#include <cuda_runtime.h>
#include <iostream>

#include "kernel.h"
#include "macro.h"

inline int divUp(int a, int b) {
  assert(b > 0);
  return ceil((float)a / b);
};

template <typename T>
__host__ __device__ __forceinline__ T bordConstant(T* src, int y, int x, int c, int h, int w,
                                                   int bordVal = 0) {
  if ((float)x >= 0 && x < w && (float)y >= 0 && y < h) {
    // printf("[%d, %d, %d, %d]", x, y, c, *(src + (y * w  + x) * 3 + c));
    return *(src + (y * w + x) * 3 + c);
  } else {
    return (T)bordVal;
  }
}

template <typename srcDtype, typename dstDtype>
__host__ __device__ __forceinline__ dstDtype linearInterp(srcDtype* src, float y, float x, int c,
                                                          int h, int w) {
  const int x1 = int(x);
  const int y1 = int(y);
  const int x2 = x1 + 1;
  const int y2 = y1 + 1;
  float out = 0;

  srcDtype src_reg = bordConstant(src, y1, x1, c, h, w);
  out = out + src_reg * ((x2 - x) * (y2 - y));

  src_reg = bordConstant(src, y1, x2, c, h, w);
  out = out + src_reg * ((x - x1) * (y2 - y));

  src_reg = bordConstant(src, y2, x1, c, h, w);
  out = out + src_reg * ((x2 - x) * (y - y1));

  src_reg = bordConstant(src, y2, x2, c, h, w);
  out = out + src_reg * ((x - x1) * (y - y1));

  return (dstDtype)out;
}

//8UC3, origin size -> 32FC3, normalized
template <bool IN_BGR, bool OUT_BGR>
__global__ void fused_preprocess_color_kernel(uint8_t* src, float* dst, int src_w, int src_h,
                                              float scale_x, float scale_y,
                                              nvocdr::ROI roi,  // roi on dst image
                                              float rbg_scale, float r_mean, float g_mean,
                                              float b_mean, float r_std, float g_std, float b_std) {
  const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

  size_t channel_size = roi.width * roi.height;
  if (dst_x < roi.width && dst_y < roi.height) {
    float fy = (float)((dst_y + roi.y + 0.5f) * scale_y - 0.5f);
    float fx = (float)((dst_x + roi.x + 0.5f) * scale_x - 0.5f);

    float b = linearInterp<uint8_t, float>(src, fy, fx, 0, src_h, src_w);
    float g = linearInterp<uint8_t, float>(src, fy, fx, 1, src_h, src_w);
    float r = linearInterp<uint8_t, float>(src, fy, fx, 2, src_h, src_w);

    if (!IN_BGR) {  // input = rgb, which mean need to swap b <-> r
      float tmp = b;
      b = r;
      r = tmp;
    }

    b = (b / rbg_scale - b_mean) / b_std;
    g = (g / rbg_scale - g_mean) / g_std;
    r = (r / rbg_scale - r_mean) / r_std;

    if (OUT_BGR) {
      *(dst + 0 * channel_size + dst_y * roi.width + dst_x) = b;
      *(dst + 1 * channel_size + dst_y * roi.width + dst_x) = g;
      *(dst + 2 * channel_size + dst_y * roi.width + dst_x) = r;
    } else {
      *(dst + 0 * channel_size + dst_y * roi.width + dst_x) = r;
      *(dst + 1 * channel_size + dst_y * roi.width + dst_x) = g;
      *(dst + 2 * channel_size + dst_y * roi.width + dst_x) = b;
    }
  }
}

template <bool IN_BGR>
__global__ void fused_preprocess_gray_kernel(uint8_t* src, float* dst, int src_w, int src_h,
                                             int dst_w, int dst_h, float gray_scale, float mean) {
  const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

  float scale_y = src_h / dst_h;
  float scale_x = src_w / dst_w;
  if (dst_x < dst_w && dst_y < dst_h) {
    float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
    float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f);

    float b = linearInterp<uint8_t, float>(src, fy, fx, 0, src_h, src_w);
    float g = linearInterp<uint8_t, float>(src, fy, fx, 1, src_h, src_w);
    float r = linearInterp<uint8_t, float>(src, fy, fx, 2, src_h, src_w);

    if (!IN_BGR) {  // input = rgb, which mean need to swap b <-> r
      float tmp = b;
      b = r;
      r = tmp;
    }

    float gray_value = (0.299 * r + 0.587 * g + 0.114 * b - mean) * gray_scale;
    *(dst + dst_y * dst_w + dst_x) = gray_value;
  }
}

namespace nvocdr {
static constexpr size_t BLOCK_SIZE_X = 32U;
static constexpr size_t BLOCK_SIZE_Y = 1024 / BLOCK_SIZE_X;

void launch_preprocess_color(uint8_t* src, float* dst, int src_w, int src_h, const cv::Rect& rect,
                             const cv::Size2f& scale, const COLOR_PREPROC_PARAM& param, bool in_bgr,
                             bool out_bgr, const cudaStream_t& stream) {
  nvocdr::ROI roi{
      .x = rect.tl().x, .y = rect.tl().y, .width = rect.size().width, .height = rect.size().height};

  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);  // 1024
  dim3 grid(divUp(roi.width, BLOCK_SIZE_X), divUp(roi.height, BLOCK_SIZE_Y));
  if (in_bgr && out_bgr) {
    fused_preprocess_color_kernel<true, true><<<grid, block, 0, stream>>>(
        src, dst, src_w, src_h, scale.width, scale.height, roi, param.rgb_scale, param.r_mean,
        param.g_mean, param.b_mean, param.r_std, param.g_std, param.b_std);
  } else if (!in_bgr && out_bgr) {
    fused_preprocess_color_kernel<false, true><<<grid, block, 0, stream>>>(
        src, dst, src_w, src_h, scale.width, scale.height, roi, param.rgb_scale, param.r_mean,
        param.g_mean, param.b_mean, param.r_std, param.g_std, param.b_std);
  } else if (in_bgr && !out_bgr) {
    fused_preprocess_color_kernel<true, false><<<grid, block, 0, stream>>>(
        src, dst, src_w, src_h, scale.width, scale.height, roi, param.rgb_scale, param.r_mean,
        param.g_mean, param.b_mean, param.r_std, param.g_std, param.b_std);
  } else {
    fused_preprocess_color_kernel<false, false><<<grid, block, 0, stream>>>(
        src, dst, src_w, src_h, scale.width, scale.height, roi, param.rgb_scale, param.r_mean,
        param.g_mean, param.b_mean, param.r_std, param.g_std, param.b_std);
  }
}

void launch_preprocess_gray(uint8_t* src, float* dst, int src_w, int src_h, int dst_w, int dst_h,
                            const GRAY_PREPROC_PARAM& param, bool in_bgr,
                            const cudaStream_t& stream) {
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);  // 1024
  dim3 grid(divUp(dst_w, BLOCK_SIZE_X), divUp(dst_h, BLOCK_SIZE_Y));
  if (in_bgr) {
    fused_preprocess_gray_kernel<true><<<grid, block, 0, stream>>>(
        src, dst, src_w, src_h, dst_w, dst_h, param.gray_scale, param.mean);
  } else {
    fused_preprocess_gray_kernel<false><<<grid, block, 0, stream>>>(
        src, dst, src_w, src_h, dst_w, dst_h, param.gray_scale, param.mean);
  }
}
}  // namespace nvocdr
