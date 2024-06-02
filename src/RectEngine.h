#ifndef __NVOCDR_RECT_HEADER__
#define __NVOCDR_RECT_HEADER__
#include <vector>
#include <algorithm>
#include <iostream>
#include "MemManager.h"
#include "utils.h"
#include "NvInfer.h"
#include "kernel.h"

using namespace nvinfer1;
#define RECT_OUTPUT_CHANNEL 1

// #define RECT_DEBUG
namespace nvocdr
{

class RectEngine
{
    public:
        RectEngine(const int& output_height, const int& output_width,const int& ocr_infer_batch, const bool upside_down = 0, const bool isNHWC=false, const float& rot_thresh=0.0, const int& rec_output_channel=1);
        ~RectEngine();

        bool initBuffer(BufferManager& buffer_mgr);
        bool setOutputBuffer(const int& index);
        bool infer(void* input_data, const Dims& input_shape, 
                   BufferManager& buffer_mgr, const std::vector<Polygon>& polys,
                   const std::vector<int>& polys_to_img, const cudaStream_t& stream = 0);
        // bool infer(BufferManager& buffer_mgr, const std::vector<std::vector<Polygon>> polys, const Dims& orig_input_shape, const cudaStream_t& stream = 0);
        void formatPoints(const Polygon& polys, std::vector<Point2d>& format_points);
        float pointDistance(const Point2d& a, const Point2d& b)
        {
            int dx = std::abs(a.x - b.x);
            int dy = std::abs(a.y - b.y);
            float dis = std::sqrt(dx*dx + dy*dy);
            return dis;
        };
        void* xmalloc(size_t size) 
        {
            void* ptr = malloc(size);
            if (ptr == NULL) {
                printf("> ERROR: malloc for size %zu failed..\n", size);
                exit(EXIT_FAILURE);
            }
            memset(ptr, 0, size);
            return ptr;
        }
        void initPtMatrix(const float *transMatrix, const int batchSize, BufferManager& buffer_mgr, const cudaStream_t& stream);
        bool solveLinearEqu(float* h_AarrayInput, float* h_BarrayInput, int n, int batchSize, BufferManager& buffer_mgr, const cudaStream_t& stream);
        void getCoefMatrix(float* coefMat, float* dstMat, std::vector<Point2d>& format_points,int w, int h);
        float det(const std::vector<float> &m);
        std::vector<float> matrixInversion(const std::vector<float>  &m);
        void warpPersceptive(void* src, const std::vector<int>& poly2Imgs, const Dims& input_shape, int outWeight, int outHeight, bool upsidedown, bool isNHWC, BufferManager& buffer_mgr,const cudaStream_t& stream);
        bool getDataFormat(){return mIsNHWC;};
        int getGrayOutputDevBufferIdx(){return mGrayOutputBufferDevIdx;};
        int getRGBOutputDevBufferIdx(){return mRGBOutputBufferDevIdx;};

        
#ifdef RECT_DEBUG
        std::string mImgSavePath;
#endif

    private:
        int mOutputBufferIndex;
        int mOutputChannel;
        int mOutputHeight;
        int mOutputWidth;
        int mOcrInferBatch;
        bool mUDFlag;
        bool mIsNHWC;
        bool mIsRGBOutput;;
        float mRotThresh = 0.0;

        // variables to save perspective trans matrix in host
        int mPtMatrixsPtrHostIdx;

        // variables to save perspective trans matrix in device
        int mPtMatrixsDevIdx;
        int mPtMatrixsPtrDevIdx;
        
        // variables for cublas in host
        cublasHandle_t mHandle;
        int mInfoArrayHostIdx;
        int mInputArrayPtrHostIdx;
        int mBarrayPtrHostIdx;

        // variables buf idx for cublas in device
        int mPloy2ImgsDevIdx;
        int mPivotArrayDevIdx;
        int mInfoArrayDevIdx;
        int mLUArrayDevIdx;
        int mBarrayDevIdx;
        int mLUArrayPtrDevIdx;
        int mBarrayPtrDevIdx;
        int mRGBOutputBufferDevIdx = -1;
        int mGrayOutputBufferDevIdx = -1;

#ifdef RECT_DEBUG
        int mRGBOutputBufferHostIdx = -1;
        int mGrayOutputBufferHostIdx = -1;
#endif

};
}
#endif