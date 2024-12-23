#ifndef __NVOCDR_CPP_HEADER__
#define __NVOCDR_CPP_HEADER__
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
#include "RectEngine.h"


// #define TRITON_DEBUG

namespace nvocdr
{
static constexpr size_t NUM_WARMUP_RUNS = 3;


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
        void process(const nvOCDRInput& input, const nvOCDROutput* output);
        // void p(const nvOCDRInput& input);


    private:
        void processTile(const nvOCDRInput& input);
        void preprocessTile(const std::vector<cv::Rect>& tiles, size_t start, size_t end);
        void postprocessTile(const std::vector<cv::Rect>& tiles, size_t start, size_t end);
        // void preprocessResize(const nvOCDR)

        
        cv::Mat mInputImage;
        cv::Mat mOCDScoreMap;
        cv::Mat mOCDValidCntMap;

        cudaStream_t mStream;
        // std::unique_ptr<OCDNetEngine> mOCDNet;
        // Dims mOCDNetInputShape;
        // int32_t mOCDNetMaxBatch;
        std::unique_ptr<OCRNetEngine> mOCRNet;
        std::unique_ptr<OCDNetEngine> mOCDNet;
        // Dims mOCRNetInputShape;
        // int32_t mOCRNetMaxBatch;
        // std::unique_ptr<RectEngine> mRect;
        // BufferManager mBuffMgr;
        nvOCDRParam mParam;
        BufferManager & mBufManager = BufferManager::Instance();

        // bool isNHWC;
};
}
#endif