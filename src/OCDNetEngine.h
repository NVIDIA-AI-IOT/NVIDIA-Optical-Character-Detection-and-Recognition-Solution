#ifndef __NVOCDR_OCDN_HEADER__
#define __NVOCDR_OCDN_HEADER__
#include <string>
#include "MemManager.h"
#include "TRTEngine.h"
#include "utils.h"
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include "kernel.h"

#define OCDNET_OUTPUT "pred"
// #define OCD_DEBUG


using namespace nvinfer1;
using namespace cv;

namespace nvocdr
{

class OCDNetEngine
{
    public:
        // @TODO(tylerz): Need to set the post-process param in the constructor
        OCDNetEngine(const std::string& engine_path, const float binaryThreshold = 0.1, const float polygonThreshold = 0.3, const float unclipRatio = 1.5, const int mMaxContourNums = 200, const bool isNHWC = true);
        ~OCDNetEngine();

        bool getIsNHWC();
        bool initTRTBuffer(BufferManager& buffer_mgr);
        bool setInputShape(const Dims& input_shape);
        bool setInputDeviceBuffer(DeviceBuffer& device_buffer, const int index);
        bool setOutputDeviceBuffer(DeviceBuffer& device_buffer, const int index);
        int32_t getMaxBatchSize() {return mEngine->getMaxBatchSize(); };
        bool preprocess(void* input_data, const Dims& input_shape, 
                        void* output_data, const Dims& output_shape, 
                        const cudaStream_t& stream = 0);
        bool infer(void* input_data, const Dims& input_shape, BufferManager& buffer_mgr,
                   std::vector<std::vector<Polygon>>& output, const cudaStream_t& stream = 0);
        
        bool postprocess(BufferManager& buffer_mgr, const Dims& input_shape, 
                         std::vector<std::vector<Polygon>>& output,
                         const cudaStream_t& stream = 0);
        float contourScore(const cv::Mat& binary, const std::vector<cv::Point>& contour);
        void unclip(const std::vector<cv::Point2f>& inPoly, std::vector<cv::Point2f> &outPoly);
        void setIsNHWC(bool order);
        void preprocessAndThresholdWarpCUDA(void* input_data, const Dims& input_shape, BufferManager& buffer_mgr, Dims& ocdOutputPatchshape,const cudaStream_t& stream);
        void preprocessWarp(void* input_data, const Dims& input_shape, BufferManager& buffer_mgr, const cudaStream_t& stream);
        void findCoutourWarp(Mat& bitmapCUDA, Mat& binary, std::vector<std::vector<Polygon>>& output, const int outputIdx);
        int getThresholdDevBufIdx()
        {
            return mOutputThresholdDevIdx;
        };
        int getOcdRawOutDevBufIdx()
        {
            return mTRTOutputBufferIndex;
        };

    private:
        std::unique_ptr<TRTEngine> mEngine;
        bool mIsNHWC;
        int mMaxContourNums = 200;
	    float mBinaryThreshold = 0.1;
        float mPolygonThreshold = 0.3;
        float mUnclipRatio = 2.0;
        float mScaleHeight = 1.0;
        float mScaleWidth = 1.0;
        
        int mTRTInputBufferIndex;
        int mTRTOutputBufferIndex;
        int mOutputThresholdHostIdx;
        int mOutputThresholdDevIdx;
        int mInferOutputbufHostIdx;

        bool mIsPatchImage;
        int mOriImgWidth;
        int mOriImgHight;

#ifdef TRITON_DEBUG
        int mImgCnt = 0;
#endif

};
}
#endif
