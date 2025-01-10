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

// template <bool IN_BGR, bool OUT_BGR>
// __global__ void fused_preprocess_color_kernel(uint8_t* src, float* dst, int src_w, int src_h,
//                                               float scale_x, float scale_y,
//                                               nvocdr::ROI roi,  // roi on dst image
//                                               float rbg_scale, float r_mean, float g_mean,
//                                               float b_mean, float r_std, float g_std, float b_std) {
//   const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
//   const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

//   size_t channel_size = roi.width * roi.height;
//   if (dst_x < roi.width && dst_y < roi.height) {
//     float fy = (float)((dst_y + roi.y + 0.5f) * scale_y - 0.5f);
//     float fx = (float)((dst_x + roi.x + 0.5f) * scale_x - 0.5f);

//     float b = linearInterp<uint8_t, float>(src, fy, fx, 0, src_h, src_w);
//     float g = linearInterp<uint8_t, float>(src, fy, fx, 1, src_h, src_w);
//     float r = linearInterp<uint8_t, float>(src, fy, fx, 2, src_h, src_w);

//     if (!IN_BGR) {  // input = rgb, which mean need to swap b <-> r
//       float tmp = b;
//       b = r;
//       r = tmp;
//     }

//     b = (b / rbg_scale - b_mean) / b_std;
//     g = (g / rbg_scale - g_mean) / g_std;
//     r = (r / rbg_scale - r_mean) / r_std;

//     if (OUT_BGR) {
//       *(dst + 0 * channel_size + dst_y * roi.width + dst_x) = b;
//       *(dst + 1 * channel_size + dst_y * roi.width + dst_x) = g;
//       *(dst + 2 * channel_size + dst_y * roi.width + dst_x) = r;
//     } else {
//       *(dst + 0 * channel_size + dst_y * roi.width + dst_x) = r;
//       *(dst + 1 * channel_size + dst_y * roi.width + dst_x) = g;
//       *(dst + 2 * channel_size + dst_y * roi.width + dst_x) = b;
//     }
//   }
// }

// template <bool IN_BGR>
// __global__ void fused_preprocess_gray_kernel(uint8_t* src, float* dst, int src_w, int src_h,
//                                              int dst_w, int dst_h, float gray_scale, float mean) {
//   const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
//   const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

//   float scale_y = src_h / dst_h;
//   float scale_x = src_w / dst_w;
//   if (dst_x < dst_w && dst_y < dst_h) {
//     float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
//     float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f);

//     float b = linearInterp<uint8_t, float>(src, fy, fx, 0, src_h, src_w);
//     float g = linearInterp<uint8_t, float>(src, fy, fx, 1, src_h, src_w);
//     float r = linearInterp<uint8_t, float>(src, fy, fx, 2, src_h, src_w);

//     if (!IN_BGR) {  // input = rgb, which mean need to swap b <-> r
//       float tmp = b;
//       b = r;
//       r = tmp;
//     }

//     float gray_value = (0.299 * r + 0.587 * g + 0.114 * b - mean) * gray_scale;
//     *(dst + dst_y * dst_w + dst_x) = gray_value;
//   }
// }

template <bool IN_BGR, bool OUT_BGR, bool OUT_GRAY>
__global__ void fused_preprocess_warp_perspective_kernel(uint8_t* src, float* dst, int src_w,
                                                         int src_h, int dst_w, int dst_h,
                                                         nvocdr::COLOR_PREPROC_PARAM param,
                                                         double m1, double m2, double m3, double m4,
                                                         double m5, double m6, double m7, double m8,
                                                         double m9) {
  const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (dst_x < dst_w && dst_y < dst_h) {
    float src_x = m1 * dst_x + m2 * dst_y + m3;
    float src_y = m4 * dst_x + m5 * dst_y + m6;
    float c = m7 * dst_x + m8 * dst_y + m9;

    src_x = src_x / c;
    src_y = src_y / c;

    float b = linearInterp<uint8_t, float>(src, src_y, src_x, 0, src_h, src_w);
    float g = linearInterp<uint8_t, float>(src, src_y, src_x, 1, src_h, src_w);
    float r = linearInterp<uint8_t, float>(src, src_y, src_x, 2, src_h, src_w);

    if (!IN_BGR) {  // input = rgb, which mean need to swap b <-> r
      float tmp = b;
      b = r;
      r = tmp;
    }
    if (OUT_GRAY) {
      float gray_value = (0.299 * r + 0.587 * g + 0.114 * b - param.r_mean) * param.rgb_scale;
      *(dst + dst_y * dst_w + dst_x) = gray_value;
    } else {
      size_t channel_size = dst_w * dst_h;
      b = (b / param.rgb_scale - param.b_mean) / param.b_std;
      g = (g / param.rgb_scale - param.g_mean) / param.g_std;
      r = (r / param.rgb_scale - param.r_mean) / param.r_std;
      if (OUT_BGR) {
        *(dst + 0 * channel_size + dst_y * dst_w + dst_x) = b;
        *(dst + 1 * channel_size + dst_y * dst_w + dst_x) = g;
        *(dst + 2 * channel_size + dst_y * dst_w + dst_x) = r;
      } else {
        *(dst + 0 * channel_size + dst_y * dst_w + dst_x) = r;
        *(dst + 1 * channel_size + dst_y * dst_w + dst_x) = g;
        *(dst + 2 * channel_size + dst_y * dst_w + dst_x) = b;
      }
    }
  }
}

namespace nvocdr {
static constexpr size_t BLOCK_SIZE_X = 32U;
static constexpr size_t BLOCK_SIZE_Y = 1024 / BLOCK_SIZE_X;

// void launch_preprocess_color(uint8_t* src, float* dst, int src_w, int src_h, const cv::Rect& rect,
//                              const cv::Size2f& scale, const COLOR_PREPROC_PARAM& param, bool in_bgr,
//                              bool out_bgr, const cudaStream_t& stream) {
//   nvocdr::ROI roi{
//       .x = rect.tl().x, .y = rect.tl().y, .width = rect.size().width, .height = rect.size().height};

//   dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);  // 1024
//   dim3 grid(divUp(roi.width, BLOCK_SIZE_X), divUp(roi.height, BLOCK_SIZE_Y));
//   double scale_x = static_cast<double>(scale.width);
//   double scale_y = static_cast<double>(scale.height);
//   if (in_bgr && out_bgr) {
//     fused_preprocess_warp_perspective_kernel<true, true, false><<<grid, block, 0, stream>>>(src, dst,
//       src_w, src_h, roi, param, scale_x, 0., roi.x * scale_x, 0, scale_y, roi.y * scale_y,
//                      0 ,0, 1);
//   } else if (!in_bgr && out_bgr) {
//     fused_preprocess_warp_perspective_kernel<false, true, false><<<grid, block, 0, stream>>>(src, dst,
//       src_w, src_h, roi, param, scale_x, 0., roi.x * scale_x, 0, scale_y, roi.y * scale_y,
//                      0 ,0, 1);
//   } else if (in_bgr && !out_bgr) {
//     fused_preprocess_warp_perspective_kernel<true, false, false><<<grid, block, 0, stream>>>(src, dst,
//       src_w, src_h, roi, param, scale_x, 0., roi.x * scale_x, 0, scale_y, roi.y * scale_y,
//                      0 ,0, 1);
//   } else {
//     fused_preprocess_warp_perspective_kernel<false, false, false><<<grid, block, 0, stream>>>(src, dst,
//       src_w, src_h, roi, param, scale_x, 0., roi.x * scale_x, 0, scale_y, roi.y * scale_y,
//                      0 ,0, 1);
//   }
// }

// void launch_fused_preprocess_warp_perspective_gray(uint8_t *src, float* dst, int src_w, int src_h, const cv::Rect& rect, const COLOR_PREPROC_PARAM& param, const cudaStream_t& stream,
// double m1,double m2,double m3,double m4,double m5,double m6,double m7,double m8,double m9) {
//     nvocdr::ROI roi{
//       .x = rect.tl().x, .y = rect.tl().y, .width = rect.size().width, .height = rect.size().height};
//   dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);  // 1024
//   dim3 grid(divUp(roi.width, BLOCK_SIZE_X), divUp(roi.height, BLOCK_SIZE_Y));
//   fused_preprocess_warp_perspective_kernel<true, false, true><<<grid, block, 0, stream>>>(src, dst,
//       src_w, src_h, roi, param, m1, m2, m3, m4, m5, m6, m7, m8, m9);
// }

template <bool IN_BGR, bool OUT_BGR, bool OUT_GRAY>
void launch_fused_warp_perspective(uint8_t* src, float* dst, const cv::Size src_size,
                                   const cv::Size& dst_size, const COLOR_PREPROC_PARAM& param,
                                   const cudaStream_t& stream, const cv::Mat& matrix) {
  // todo, use better block size,  
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);  // 1024
  dim3 grid(divUp(dst_size.width, BLOCK_SIZE_X), divUp(dst_size.height, BLOCK_SIZE_Y));


  auto mat = reinterpret_cast<double*>(matrix.data);

  fused_preprocess_warp_perspective_kernel<IN_BGR, OUT_BGR, OUT_GRAY><<<grid, block, 0, stream>>>(
      src, dst, src_size.width, src_size.height, dst_size.width, dst_size.height, param, mat[0],
      mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8]);
}
template void launch_fused_warp_perspective<true, true, false>(
    uint8_t* src, float* dst, const cv::Size src_size, const cv::Size& dst_size,
    const COLOR_PREPROC_PARAM& param, const cudaStream_t& stream, const cv::Mat& matrix);
template void launch_fused_warp_perspective<true, false, false>(
    uint8_t* src, float* dst, const cv::Size src_size, const cv::Size& dst_size,
    const COLOR_PREPROC_PARAM& param, const cudaStream_t& stream, const cv::Mat& matrix);
template void launch_fused_warp_perspective<false, true, false>(
    uint8_t* src, float* dst, const cv::Size src_size, const cv::Size& dst_size,
    const COLOR_PREPROC_PARAM& param, const cudaStream_t& stream, const cv::Mat& matrix);
template void launch_fused_warp_perspective<false, false, false>(
    uint8_t* src, float* dst, const cv::Size src_size, const cv::Size& dst_size,
    const COLOR_PREPROC_PARAM& param, const cudaStream_t& stream, const cv::Mat& matrix);

template void launch_fused_warp_perspective<true, false, true>(
    uint8_t* src, float* dst, const cv::Size src_size, const cv::Size& dst_size,
    const COLOR_PREPROC_PARAM& param, const cudaStream_t& stream, const cv::Mat& matrix);

template void launch_fused_warp_perspective<false, false, true>(
    uint8_t* src, float* dst, const cv::Size src_size, const cv::Size& dst_size,
    const COLOR_PREPROC_PARAM& param, const cudaStream_t& stream, const cv::Mat& matrix);
}  // namespace nvocdr
