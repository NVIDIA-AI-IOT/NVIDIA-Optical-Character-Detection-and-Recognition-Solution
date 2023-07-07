#ifndef NVOCDR_KERNEL_HEADER
#define NVOCDR_KERNEL_HEADER
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cassert>
#include "NvInfer.h"

#define CUDA_BLOCK_THREADS 1024
#define BLOCK 32
#define PERSPECTIVE_TRANSFORMATION_MATRIX_DIM 8
#define IMG_MEAN_GRAY 127.5
#define IMG_SCALE_GRAY 0.00784313
#define IMG_MEAN_B  104.00698793
#define IMG_MEAN_G  116.66876762
#define IMG_MEAN_R  122.67891434
#define IMG_SCALE_BRG  0.00392156
#define IMG_PIXEL_MAXVAL  255
#define CV_DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))
#define GRAY_SHIFT  15
#define RY15  9798  // == 0.299f*32768 + 0.5
#define GY15  19235 // == 0.587f*32768 + 0.5
#define BY15  3735  // == 0.114f*32768 + 0.5
#define PI    3.1415926535897932384626433832795

// tl and bt points coords (4) and the area (1), and angle 1
#define MIN_AREA_EACH_ANGLE_STRID 6
#define ROTATE_DEGREES 90
typedef unsigned char uchar;


#ifndef checkKernelErrors
#    define checkKernelErrors(expr)                                                               \
        do                                                                                        \
        {                                                                                         \
            expr;                                                                                 \
                                                                                                  \
            cudaError_t __err = cudaGetLastError();                                               \
            if (__err != cudaSuccess)                                                             \
            {                                                                                     \
                printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err)); \
                abort();                                                                          \
            }                                                                                     \
        }                                                                                         \
        while (0)
#endif

// CUDA Runtime error messages
static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)


struct ImgROI
{
  __host__ __device__ __forceinline__ ImgROI()
  {

  }
  __host__ __device__ __forceinline__ ImgROI(int _x, int _y, int _w, int _h)
  :x(_x)
  ,y(_y)
  ,w(_w)
  ,h(_h)
  {

  }

  // xy is the topleft point, wh is the width and height of the ROI
  int  x;
  int  y;
  int  w;
  int  h;
};

template<typename T>
struct ImagePtrCUDA
{

    __host__ __device__ __forceinline__ ImagePtrCUDA()
    {

    }

    __host__ __device__ __forceinline__ ImagePtrCUDA(int batches_, int rows_, int cols_, int ch_, T *data_, bool isNHWC_=true)
        : batches(batches_)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , data(data_)
        , isNHWC(isNHWC_)
    {
        imgStride = rows * cols * ch;
        if(isNHWC)
        {
            rowStride = cols * ch;
            chStride = 1;
        }
        else
        {
            rowStride = cols;
            chStride = rowStride * rows;
        }
    }

    // ptr for uchar, ushort, float, typename T -> uchar etc.
    // each fetch operation get a single channel element
    __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c)
    {
        if(isNHWC)
        {
            return &data[b * imgStride + y * rowStride + x * ch + c];
        }
        else
        {
            return &data[b * imgStride + c * chStride + y * rowStride + x];
        }
    }

    bool isNHWC;
    int  batches;
    int  rows;
    int  cols;
    int  ch;
    int  imgStride;
    int  rowStride;
    int  chStride;
    T    *data;
};


template<typename T>
__host__ __device__ __forceinline__ T bordConstant(ImagePtrCUDA<T> src, int b,  int y, int x, int c, int bordVal)
{
    if((float)x >= 0 && x < src.cols && (float)y >= 0 && y < src.rows)
    {
        return *src.ptr(b, y, x, c);
    }
    else 
    {
        return (T)bordVal;
    }
}

template<typename srcDtype, typename dstDtype>
__host__ __device__ __forceinline__ dstDtype linearInterp(ImagePtrCUDA<srcDtype>& src, int bidx, float y, float x, int c)
{
    const int x1 = int(x);
    const int y1 = int(y);
    const int x2 = x1 + 1;
    const int y2 = y1 + 1;
    float out = 0;

    srcDtype src_reg = bordConstant(src, bidx, y1, x1, c, 0);
    out               = out + src_reg * ((x2 - x) * (y2 - y));

    src_reg = bordConstant(src, bidx, y1, x2, c, 0);
    out     = out + src_reg * ((x - x1) * (y2 - y));

    src_reg = bordConstant(src, bidx, y2, x1, c, 0);
    out     = out + src_reg * ((x2 - x) * (y - y1));

    src_reg = bordConstant(src, bidx, y2, x2, c, 0);
    out     = out + src_reg * ((x - x1) * (y - y1));
    
    return (dstDtype)out;
}

template<typename T>
struct ContourPtrCUDA
{

    __host__ __device__ __forceinline__ ContourPtrCUDA()
    {
    }

    __host__ __device__ __forceinline__ ContourPtrCUDA(int maxNumPointsIncontour_, int pointDim_, T *data_)
        : maxNumPointsIncontour(maxNumPointsIncontour_)
        , pointDim(pointDim_)
        , data(data_)
    {
        contourStride = maxNumPointsIncontour * pointDim;
        pointStride = pointDim;
    }

    __host__ __device__ __forceinline__ T *ptr(int coutourIdx, int pointIdx, int dimIdx)
    {
        return &data[coutourIdx * contourStride + pointIdx * pointStride + dimIdx];
    }


    int  maxNumPointsIncontour;
    int  pointDim;
    int  contourStride;
    int  pointStride;
    T    *data;
};

template<typename T>
struct MinAreaRectPtrCUDA
{
    __host__ __device__ __forceinline__ MinAreaRectPtrCUDA()
    {
    }

    __host__ __device__ __forceinline__ MinAreaRectPtrCUDA(int maxNumRect_, int maxAngle_, T *rectData_)
        : maxNumRect(maxNumRect_)
        , maxAngle(maxAngle_)
        , rectData(rectData_)
    {
      
      angleStride = MIN_AREA_EACH_ANGLE_STRID;
      rectStride = maxAngle * angleStride;
    }
    
    __host__ __device__ __forceinline__ T *pointPtr(int rectIdx, int angle, int p)
    {
        return &rectData[rectIdx * rectStride + angle * angleStride + p];
    }

    __host__ __device__ __forceinline__ T *minAreaPointPtr(int rectIdx, int pointIdx, int coord)
    {
        return &rectData[maxNumRect * rectStride + rectIdx*8 + pointIdx*2 + coord];
    }

    int  maxNumRect;
    int  maxAngle;
    int  rectStride;
    int  angleStride;
    T    *rectData;
};

//In-place scale and substract
void subscal(const unsigned int sample_cnt, void* data, float scale, float mean, const cudaStream_t& stream = 0);

void warp_caller(const ImagePtrCUDA<uchar>& src, ImagePtrCUDA<uchar>& dst, ImagePtrCUDA<float>& dst_gray, float** ptMatrixPtr, int* poly2Imgs, bool upsidedown, const cudaStream_t& stream);

// ImagePtrCUDA<uchar> transpose_launcher( void* data, const int batchSize, const int w, const int h, const int c, bool isNHWC);
void blobFromImagesCUDA(void* inData, void* outData, const nvinfer1::Dims& inShape, const nvinfer1::Dims& outShape, bool inputIsNHWC, const cudaStream_t& stream);

void thresholdCUDA(void* inData, void* outData, const nvinfer1::Dims& inShape, const float binaryThreshold, const cudaStream_t& stream);

void minAreaRectCUDA(ContourPtrCUDA<uint16_t>& inContourPointsData, MinAreaRectPtrCUDA<int>& outMinAreaRectBox, void* rotateCoeffs, void* numPointsInContour, void* contoursToImages, const int numContours, const int maxNumPointsInContour, const int numOfDegrees, const cudaStream_t& stream);

void calculateRotateCoefCUDA(void* rotateCoefBuf, const int degrees ,const cudaStream_t& stream);

void patchMergeWarp(void* patchData, void* mergeData, void* patchOcdOutRawData, void* mergeOcdOutRawData, const nvinfer1::Dims& patchShape, const nvinfer1::Dims& mergeShape, const ImgROI& patchROI, const ImgROI& mergeROI,const cudaStream_t& stream);

float KeepAspectRatioResize(void* inData, void* outData, const nvinfer1::Dims& inShape, const int32_t out_h, const int32_t out_w, const cudaStream_t& stream);

#endif