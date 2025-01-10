#ifndef __NVOCDR_CPP_HEADER__
#define __NVOCDR_CPP_HEADER__
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvocdr.h"
#include "MemManager.h"
#include "OCRNetNvCLIP4STREngine.h"
#include "OCRNetEngine.h"
#include "OCDNetEngine.h"
#include "RectEngine.h"

// #define TRITON_DEBUG

namespace nvocdr
{

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
        nvOCDR(nvOCDRParam param);
        ~nvOCDR();

        // inferV1:
        // - OCDNet - postprocess: CPU
        // - Rect CPU
        // - OCRNet - postprocess: CPU
        void inferV1(void* input_data, const Dims& input_shape,
                     std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>>& output);

        // infer:
        // wrap inferV1 to process input batch size > max batch size of trt engine
        void infer(void* input_data, const Dims& input_shape,
                     std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>>& output);

        void inferPatch(void* oriImgData, void* patchImgs, const Dims& input_shape, const Dims& oriImgshape,const float overlapRate,
                        std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>>& output);
        void wrapInferPatch(void* input_data, const Dims& input_shape, float overlap_rate,
                            std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>>& output);
        void OCRNetInferWarp(void* input_data, const Dims& input_shape, std::vector<std::vector<Polygon>>& polys, std::vector<std::vector<std::pair<Polygon, std::pair<std::string, float>>>>& output);
        void patchMaskMergeCUDA(void* oriThresholdData, void* patchThresholdData, void* oriRawData, void* patchRawData, const Dims& patchImgsShape, const Dims& oriImgshape, const float overlapRate, const int col_idx, const int row_idx, const int num_col_cut, const int num_row_cut ,const cudaStream_t& stream);

    private:
        bool paramCheck();

        cudaStream_t mStream;
        std::unique_ptr<OCDNetEngine> mOCDNet;
        Dims mOCDNetInputShape;
        int32_t mOCDNetMaxBatch;
        // std::unique_ptr<OCRNetEngine> mOCRNet;
        std::unique_ptr<OCRNetBaseEngine> mOCRNet;
        Dims mOCRNetInputShape;
        int32_t mOCRNetMaxBatch;
        std::unique_ptr<RectEngine> mRect;
        BufferManager mBuffMgr;
        nvOCDRParam mParam;
        bool isNHWC;
};
}
#endif