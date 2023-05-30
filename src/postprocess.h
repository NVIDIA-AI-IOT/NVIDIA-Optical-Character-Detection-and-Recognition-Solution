#ifndef __POSTPROCESS_HEADER__
#define __POSTPROCESS_HEADER__
#include <string>
#include "utils.h"
#include "RectEngine.h"
#include "kernel.h"

namespace nvocdr
{
struct BoxInfo
{
    Box2d box;
    Point2d_f boxCenter;
    Point2d_f leftSideCenter;
    Point2d_f rightSideCenter;
};
struct Letter
{
    int id;
    std::string text;
    Polygon polys;
    BoxInfo boxInfo;
};

class Letter2Sentence
{
public:
    Letter2Sentence(){
        
        cudaStreamCreate(&mStream);
        initBuffer();
    };
    ~Letter2Sentence(){};

    bool initBuffer();
    void getBoxInfo(const Polygon& polys, BoxInfo& boxInfo);
    void extractSentence(std::vector<std::string>& texts, std::vector<std::vector<int>>& boxes, std::vector<std::string>& sentence);
    void generateSentence(const int keyIdx, std::vector<Letter>& letters, std::vector<int>& letterIdxList, float* sideDisMetrix, std::vector<int>& sentence);

private:
    int mCenterHostBufIdx;
    int mCenterDevBufIdx;
    int mLeftSideCenterHostBufIdx;
    int mLeftSideCenterDevBufIdx;
    int mRightSideCenterDevBufIdx;
    int mRightSideCenterHostBufIdx;
    int mLetterMaskDevBufIdx;
    int mLetterMaskHostBufIdx;
    int mR2LDisMetrixrDevBufIdx;
    int mR2LDisMetrixrDevHostIdx;
    BufferManager mBuffer_mgr;
    cudaStream_t mStream;
};

}
#endif
