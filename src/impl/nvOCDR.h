#pragma once
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <cuda.h>
#include <cuda_runtime.h>

#include "opencv2/opencv.hpp"
#include "nvocdr.h"
#include "MemManager.h"
#include "OCRNetEngine.h"
#include "OCDNetEngine.h"

namespace nvocdr
{
static constexpr float IMG_MEAN_GRAY = 127.5;
static constexpr float IMG_SCALE_GRAY = 0.00784313;
static constexpr float IMG_MEAN_B = 104.00698793;
static constexpr float IMG_MEAN_G = 116.66876762;
static constexpr float IMG_MEAN_R = 122.67891434;
static constexpr float IMG_SCALE_BRG = 0.00392156;

static constexpr size_t NUM_WARMUP_RUNS = 10;
static constexpr size_t QUAD = 4;


// static constexpr size_t TRIA = 3;


//nvOCDR wrap the whole pipeline
//version 0:
// - single-thread
// - single cuda stream
// - uint8_t input / float output
// - NHWC input data format
// - English word only
class nvOCDR
{
    public:
        nvOCDR(const nvOCDRParam& param);
        // ~nvOCDR();

        // inferV1:
        // - OCDNet - postprocess: CPU
        // - Rect CPU
        // - OCRNet - postprocess: CPU
        // void inferV1(void* input_data, const Dims& input_shape,
        //              std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>>& output);

        // // infer:
        // // wrap inferV1 to process input batch size > max batch size of trt engine
        // void infer(void* input_data, const Dims& input_shape,
        //              std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>>& output);

        // void inferPatch(void* oriImgData, void* patchImgs, const Dims& input_shape, const Dims& oriImgshape,const float overlapRate,
        //                 std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>>& output);
        // void wrapInferPatch(void* input_data, const Dims& input_shape, float overlap_rate,
        //                     std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>>& output);
        // void OCRNetInferWarp(void* input_data, const Dims& input_shape, std::vector<std::vector<Polygon>>& polys, std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>>& output);
        // void patchMaskMergeCUDA(void* oriThresholdData, void* patchThresholdData, void* oriRawData, void* patchRawData, const Dims& patchImgsShape, const Dims& oriImgshape, const float overlapRate, const int col_idx, const int row_idx, const int num_col_cut, const int num_row_cut ,const cudaStream_t& stream);

        // void getResults(const nvOCDRInput& input, nvOCDROutputMeta * const output);
        // void addInput(const nvOCDRInput& input);
        // void getOuput(const nvOCDROutputMeta* output);
        void process(const nvOCDRInput& input,  nvOCDROutput* const output);
        // void p(const nvOCDRInput& input);


    private:
        void processTile(const nvOCDRInput& input);

        void handleStrategy(const nvOCDRInput& input);

        void getTilePlan(size_t input_w, size_t input_h, size_t raw_w, size_t raw_h, size_t stride);
        void preprocessOCDTile(size_t start, size_t end);
        void postprocessOCDTile(size_t start, size_t end);

        void selectOCRCandidates();
        void preprocessOCR(size_t start, size_t end, size_t bl_pt_idx);
        void postprocessOCR(size_t start, size_t end);

        void preprocessInputImage();
        void restoreImage(const nvOCDRInput &input);
        void setOutput(nvOCDROutput* const output);
        cv::Mat denormalizeGray(const cv::Mat& input);

        std::unique_ptr<OCRNetEngine> mOCRNet;
        std::unique_ptr<OCDNetEngine> mOCDNet;
        BufferManager & mBufManager = BufferManager::Instance();
        
        cv::Mat mInputImage; // origin input image, aribitrial size
        cv::Mat mInputImageResized; // origin input image, aribitrial size

        cv::Mat mInputImageResized32F; // input image, float32, resized
        cv::Mat mOCDScoreMap; // OCD score map, size = resized 
        cv::Mat mOCDOutputMask; // OCD output mask, size = resized
        cv::Mat mOCDValidCntMap; // OCD valid cnt map, size = resized

        cv::Mat mInputGrayImage; // input image, float32 and gray, size = mInputImage

        /** origin size, resized size */
        std::pair<cv::Size, cv::Size> mResizeInfo;
        std::vector<cv::Rect> mTiles;
        
        std::vector<std::vector<cv::Point>> mTextCntrCandidates;

        std::vector<std::array<cv::Point2f, QUAD>> mQuadPts; 
        
        std::vector<Text> mTexts;
        size_t mNumTexts;

        cudaStream_t mStream;
        nvOCDRParam mParam;
};
}
