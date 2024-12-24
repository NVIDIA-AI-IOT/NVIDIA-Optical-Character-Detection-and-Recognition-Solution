#pragma once

#include <string>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "MemManager.h"
#include "TRTEngine.h"
#include "kernel.h"
#include "nvocdr.h"

// #define OCD_DEBUG

namespace nvocdr
{
constexpr char OCD_PREFIX[] = "OCD";
constexpr char OCDNET_INPUT[] = "input";
constexpr char OCDNET_OUTPUT[] = "pred";

class OCDNetEngine : public OCDTRTEngine
{
    public:
        bool customInit() final;
        // OCRNetEngine() = default;
        OCDNetEngine(const char name[], const nvOCDParam& param) : OCDTRTEngine(name, param) { };
        // void process(const cudaStream_t& stream);
        // void postprocess(const cudaStream_t& stream);

        

        // @TODO(tylerz): Need to set the post-process param in the constructor
        // OCDNetEngine(const std::string& engine_path, const float binaryThreshold = 0.1, const float polygonThreshold = 0.3, const float unclipRatio = 1.5, const int mMaxContourNums = 200, const bool isNHWC = true);
        // ~OCDNetEngine();

        // // bool getIsNHWC();
        // bool initTRTBuffer(BufferManager& buffer_mgr);
        // bool setInputShape(const Dims& input_shape);
        // bool setInputDeviceBuffer(DeviceBuffer& device_buffer, const int index);
        // bool setOutputDeviceBuffer(DeviceBuffer& device_buffer, const int index);
        // int32_t getMaxBatchSize() {return mEngine->getMaxBatchSize(); };
        // bool preprocess(void* input_data, const Dims& input_shape, 
        //                 void* output_data, const Dims& output_shape, 
        //                 const cudaStream_t& stream = 0);
        // bool infer(void* input_data, const Dims& input_shape, BufferManager& buffer_mgr,
        //            std::vector<std::vector<Polygon>>& output, const cudaStream_t& stream = 0);
        
        // bool postprocess(BufferManager& buffer_mgr, const Dims& input_shape, 
        //                  std::vector<std::vector<Polygon>>& output,
        //                  const cudaStream_t& stream = 0);
        // float contourScore(const cv::Mat& binary, const std::vector<cv::Point>& contour);
        // void unclip(const std::vector<cv::Point2f>& inPoly, std::vector<cv::Point2f> &outPoly);
        // // void setIsNHWC(bool order);
        // void preprocessAndThresholdWarpCUDA(void* input_data, const Dims& input_shape, BufferManager& buffer_mgr, Dims& ocdOutputPatchshape,const cudaStream_t& stream);
        // void preprocessWarp(void* input_data, const Dims& input_shape, BufferManager& buffer_mgr, const cudaStream_t& stream);
        // void findCoutourWarp(cv::Mat& bitmapCUDA, cv::Mat& binary, std::vector<std::vector<Polygon>>& output, const int outputIdx);
        // size_t getThresholdDevBufIdx()
        // {
        //     return mOutputThresholdDevIdx;
        // };
        // size_t getOcdRawOutDevBufIdx()
        // {
        //     return mTRTOutputBufferIndex;
        // };

    private:
        // nvOCDParam mParam;
        // // std::unique_ptr<TRTEngine> mEngine;
        // // bool mIsNHWC;
        // // DataFormat mDataFormat;
        // // int mMaxContourNums = 200;
	    // // float mBinaryThreshold = 0.1;
        // // float mPolygonThreshold = 0.3;
        // // float mUnclipRatio = 2.0;
        // // float mScaleHeight = 1.0;
        // // float mScaleWidth = 1.0;
        
        // size_t mTRTInputBufferIndex;
        // size_t mTRTOutputBufferIndex;
        // size_t mOutputThresholdHostIdx;
        // size_t mOutputThresholdDevIdx;
        // size_t mInferOutputbufHostIdx;

        // bool mIsPatchImage;
        // size_t mOriImgWidth;
        // size_t mOriImgHight;
};
}
