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
static constexpr size_t NUM_WARMUP_RUNS = 10;
static constexpr size_t QUAD = 4;


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

        void preprocessOCDTile(size_t start, size_t end);
        void postprocessOCDTile(size_t start, size_t end);

        void procesOCR();
        void selectOCRCandidates();
        void preprocessOCR(size_t start, size_t end);
        void postprocessOCR(size_t start, size_t end);

        std::unique_ptr<OCRNetEngine> mOCRNet;
        std::unique_ptr<OCDNetEngine> mOCDNet;
        BufferManager & mBufManager = BufferManager::Instance();
        
        cv::Mat mInputImage; // origin input image

        cv::Mat mInputImage32F; // input image, float32 format
        cv::Mat mInputGrayImage; // input image, float32 and gray 
        cv::Mat mOCDScoreMap;
        cv::Mat mOCDOutputMask;
        cv::Mat mOCDValidCntMap;

        std::vector<cv::Rect> mTiles;
        std::vector<std::vector<cv::Point>> mTextCntrCandidates;
        std::vector<std::array<cv::Point2f, QUAD>> mQuadPts; 
        std::vector<Text> mTexts;
        size_t mNumTexts;

        cudaStream_t mStream;
        nvOCDRParam mParam;
};
}
