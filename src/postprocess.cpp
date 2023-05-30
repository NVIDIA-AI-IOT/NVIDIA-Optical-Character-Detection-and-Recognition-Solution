#include <iostream>
#include <set>
#include "postprocess.h"

namespace nvocdr
{

float pointDistance(const Point2d& a, const Point2d& b)
{
    int dx = std::abs(a.x - b.x);
    int dy = std::abs(a.y - b.y);
    float dis = std::sqrt(dx*dx + dy*dy);
    return dis;
}

void formatPoints(const Polygon& polys, std::vector<Point2d>& format_points)
{
    if(format_points.size() != 4)
    {
        return;
    }
    std::vector<Point2d> points{Point2d{polys.x1,polys.y1}, Point2d{polys.x2,polys.y2},
                                Point2d{polys.x3,polys.y3}, Point2d{polys.x4,polys.y4}};
    std::sort(points.begin(), points.end(), [](const Point2d& a, const Point2d& b){return a.x < b.x;});

    if(points[0].y <= points[1].y)
    {
        format_points[0] = points[0];
        format_points[3] = points[1];
    }
    else
    {
        format_points[0] = points[1];
        format_points[3] = points[0];
    }
    
    if(points[2].y <= points[3].y)
    {
        format_points[1] = points[2];
        format_points[2] = points[3];
    }
    else
    {
        format_points[1] = points[3];
        format_points[2] = points[2];
    }

    // a vecotr to save distance, in order by topside, leftside
    std::vector<float> distances(4);
    distances[0] = pointDistance(format_points[0], format_points[1]);
    distances[1] = pointDistance(format_points[0], format_points[3]);
    if(distances[1] > distances[0])
    {
        Point2d tmp_point = format_points[0];
        format_points[0] = format_points[3];
        format_points[3] = format_points[2];
        format_points[2] = format_points[1];
        format_points[1] = tmp_point;
    }

    return;
}

void Letter2Sentence::getBoxInfo(const Polygon& polys, BoxInfo& boxInfo)
{
    std::vector<Point2d> format_points(4);
    formatPoints(polys, format_points);
    boxInfo.box.leftTop     = format_points[0];
    boxInfo.box.rightTop    = format_points[1];
    boxInfo.box.rightBottom = format_points[2];
    boxInfo.box.leftBottom  = format_points[3];
    boxInfo.boxCenter.x = float((boxInfo.box.leftBottom.x + boxInfo.box.leftTop.x + boxInfo.box.rightBottom.x + boxInfo.box.rightTop.x)/4.0);
    boxInfo.boxCenter.y = float((boxInfo.box.leftBottom.y + boxInfo.box.leftTop.y + boxInfo.box.rightBottom.y + boxInfo.box.rightTop.y)/4.0);
    boxInfo.leftSideCenter.x = float(boxInfo.box.leftTop.x + boxInfo.box.leftBottom.x)/2.0;
    boxInfo.leftSideCenter.y = float(boxInfo.box.leftTop.y + boxInfo.box.leftBottom.y)/2.0;
    boxInfo.rightSideCenter.x = float(boxInfo.box.rightTop.x + boxInfo.box.rightBottom.x)/2.0;
    boxInfo.rightSideCenter.y = float(boxInfo.box.rightTop.y + boxInfo.box.rightBottom.y)/2.0;
    return;
}

bool Letter2Sentence::initBuffer()
{
    
    mCenterHostBufIdx = mBuffer_mgr.initHostBuffer(MAX_LETTERS_IN_IMAGE*2, sizeof(float));
    mCenterDevBufIdx = mBuffer_mgr.initDeviceBuffer(MAX_LETTERS_IN_IMAGE*2, sizeof(float));
    mLeftSideCenterHostBufIdx = mBuffer_mgr.initHostBuffer(MAX_LETTERS_IN_IMAGE*2, sizeof(float));
    mLeftSideCenterDevBufIdx = mBuffer_mgr.initDeviceBuffer(MAX_LETTERS_IN_IMAGE*2, sizeof(float));
    mRightSideCenterDevBufIdx = mBuffer_mgr.initDeviceBuffer(MAX_LETTERS_IN_IMAGE*2, sizeof(float));
    mRightSideCenterHostBufIdx = mBuffer_mgr.initHostBuffer(MAX_LETTERS_IN_IMAGE*2, sizeof(float));
    mLetterMaskDevBufIdx = mBuffer_mgr.initDeviceBuffer(MAX_LETTERS_IN_IMAGE*MAX_LETTERS_IN_IMAGE, sizeof(short));
    mLetterMaskHostBufIdx = mBuffer_mgr.initHostBuffer(MAX_LETTERS_IN_IMAGE*MAX_LETTERS_IN_IMAGE, sizeof(short));
    mR2LDisMetrixrDevBufIdx = mBuffer_mgr.initDeviceBuffer(MAX_LETTERS_IN_IMAGE*MAX_LETTERS_IN_IMAGE, sizeof(float));
    mR2LDisMetrixrDevHostIdx = mBuffer_mgr.initHostBuffer(MAX_LETTERS_IN_IMAGE*MAX_LETTERS_IN_IMAGE, sizeof(float));
}

void Letter2Sentence::generateSentence(const int keyIdx, std::vector<Letter>& letters, std::vector<int>& letterIdxList, float* sideDisMetrix, std::vector<int>& sentence)
{
    if (letterIdxList.size() <2 )
    {
        sentence.push_back(keyIdx);
        return;
    }
    std::vector<std::vector<int>> maybeSentence;
    for (int i=0; i<letterIdxList.size(); i++)
    {
        bool alreadyIn = false;
        for (int j=i+1; j<letterIdxList.size(); j++)
        {
           
            if(sideDisMetrix[letterIdxList[i]* MAX_LETTERS_IN_IMAGE + letterIdxList[j]]< SIDE_DISTANCE_THRESHOLD)
            {
                for (int mayStcIdx=0 ; mayStcIdx<maybeSentence.size(); mayStcIdx++)
                {
                    if (std::find(maybeSentence[mayStcIdx].begin(), maybeSentence[mayStcIdx].end(), letterIdxList[i]) != maybeSentence[mayStcIdx].end())
                    { 
                        maybeSentence[mayStcIdx].push_back(letterIdxList[j]);
                        alreadyIn = true;
                        break;
                    }
                }
                if (! alreadyIn)
                {
                    std::vector<int> tmpStn{letterIdxList[i], letterIdxList[j]};
                    maybeSentence.emplace_back(tmpStn);
                }
            }        
        } 
    }

    if (maybeSentence.size()!=0)
    {
        for (auto stn: maybeSentence)
        {
            if(std::find(stn.begin(), stn.end(), keyIdx) != stn.end())
            {
                for(auto idx:stn)
                {
                    sentence.emplace_back(idx);
                }
                return ;
            }   
        } 
    }
    sentence.emplace_back(keyIdx);
    return ;
}

void Letter2Sentence::extractSentence(std::vector<std::string>& texts, std::vector<std::vector<int>>& boxes, std::vector<std::string>& sentence)
{

    float* centerHostBuf = static_cast<float*>(mBuffer_mgr.mHostBuffer[mCenterHostBufIdx].data());
    float* leftCenterHostBuf = static_cast<float*>(mBuffer_mgr.mHostBuffer[mLeftSideCenterHostBufIdx].data());
    float* rightCenterHostBuf = static_cast<float*>(mBuffer_mgr.mHostBuffer[mRightSideCenterHostBufIdx].data());
    int letterCnt = texts.size();
    std::vector<Letter> letters;
    for (int i=0; i<letterCnt; i++)
    {
        // Letter letter = wordsInfo[i];
        Letter letter;
        letter.text = texts[i];
        letter.polys.x1 = boxes[i][1]; letter.polys.y1 = boxes[i][2];
        letter.polys.x2 = boxes[i][3]; letter.polys.y2 = boxes[i][4];
        letter.polys.x3 = boxes[i][5]; letter.polys.y3 = boxes[i][6];
        letter.polys.x4 = boxes[i][7]; letter.polys.y4 = boxes[i][8];
        letters.push_back(letter);
        getBoxInfo(letter.polys,letter.boxInfo);
        centerHostBuf[2*i] = letter.boxInfo.boxCenter.x;
        centerHostBuf[2*i+1] = letter.boxInfo.boxCenter.y;
        leftCenterHostBuf[2*i] = letter.boxInfo.leftSideCenter.x;
        leftCenterHostBuf[2*i+1] = letter.boxInfo.leftSideCenter.y;
        rightCenterHostBuf[2*i] = letter.boxInfo.rightSideCenter.x;
        rightCenterHostBuf[2*i+1] = letter.boxInfo.rightSideCenter.y;
    }

    checkCudaErrors(cudaMemcpyAsync(mBuffer_mgr.mDeviceBuffer[mCenterDevBufIdx].data(), mBuffer_mgr.mHostBuffer[mCenterHostBufIdx].data(), (letterCnt*2)*sizeof(float), cudaMemcpyHostToDevice, mStream));
    checkCudaErrors(cudaMemcpyAsync(mBuffer_mgr.mDeviceBuffer[mLeftSideCenterDevBufIdx].data(), mBuffer_mgr.mHostBuffer[mLeftSideCenterHostBufIdx].data(), (letterCnt*2)*sizeof(float), cudaMemcpyHostToDevice, mStream));
    checkCudaErrors(cudaMemcpyAsync(mBuffer_mgr.mDeviceBuffer[mRightSideCenterDevBufIdx].data(), mBuffer_mgr.mHostBuffer[mRightSideCenterHostBufIdx].data(), (letterCnt*2)*sizeof(float), cudaMemcpyHostToDevice, mStream));
    // auto m_beg = std::chrono::high_resolution_clock::now();
    calculateBoxDistance(mBuffer_mgr.mDeviceBuffer[mCenterDevBufIdx].data(), mBuffer_mgr.mDeviceBuffer[mLeftSideCenterDevBufIdx].data(), mBuffer_mgr.mDeviceBuffer[mRightSideCenterDevBufIdx].data(), 
                        mBuffer_mgr.mDeviceBuffer[mLetterMaskDevBufIdx].data(), mBuffer_mgr.mDeviceBuffer[mR2LDisMetrixrDevBufIdx].data(), letterCnt, mStream);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - m_beg);
    // std::cout<< "letter2sentence CUDA time " << dur.count() << " us" << std::endl;
    checkCudaErrors(cudaMemcpyAsync(mBuffer_mgr.mHostBuffer[mLetterMaskHostBufIdx].data(), mBuffer_mgr.mDeviceBuffer[mLetterMaskDevBufIdx].data(), mBuffer_mgr.mDeviceBuffer[mLetterMaskDevBufIdx].nbBytes(), cudaMemcpyDeviceToHost, mStream));
    checkCudaErrors(cudaMemcpyAsync(mBuffer_mgr.mHostBuffer[mR2LDisMetrixrDevHostIdx].data(), mBuffer_mgr.mDeviceBuffer[mR2LDisMetrixrDevBufIdx].data(), mBuffer_mgr.mDeviceBuffer[mR2LDisMetrixrDevBufIdx].nbBytes(), cudaMemcpyDeviceToHost, mStream));

    short* letterMaskHost = static_cast<short*>(mBuffer_mgr.mHostBuffer[mLetterMaskHostBufIdx].data());
    float* r2LDisMetrixrHost = static_cast<float*>(mBuffer_mgr.mHostBuffer[mR2LDisMetrixrDevHostIdx].data());
    std::vector<std::pair<int, std::vector<int>>> candidateLetters;
    for(int i=0; i< letterCnt; i++)
    {
        std::vector<int> candidate;
        for(int j=0; j<letterCnt; j++)
        {
            if( letterMaskHost[i*MAX_LETTERS_IN_IMAGE + j]  == 1)
            {
                candidate.push_back(j);
            }
        }
        std::sort(candidate.begin(), candidate.end(), [centerHostBuf](int a, int b){return centerHostBuf[2*a] < centerHostBuf[2*b]; } );
        candidateLetters.emplace_back(std::make_pair(i,candidate));

    }

    std::set<int> alreadUsedLetter;
    std::vector<std::vector<int>> allSentences;
    for(auto candidate: candidateLetters)
    {
        int keyIdx = candidate.first;
        if (alreadUsedLetter.count(keyIdx) !=0 )
        {
            continue;
        }
        auto candidateVec = candidate.second;
        std::vector<int> sentenceLetterIdx;
        generateSentence(keyIdx, letters, candidateVec,r2LDisMetrixrHost, sentenceLetterIdx);
        for(auto s : sentenceLetterIdx)
        {
            alreadUsedLetter.insert(s);
        }
        allSentences.emplace_back(sentenceLetterIdx);
    }

    std::sort(allSentences.begin(), allSentences.end(), [centerHostBuf](std::vector<int>& a, std::vector<int>& b){return centerHostBuf[2*a[0]+1] < centerHostBuf[2*b[0]+1]; } );

    for(auto sentenceVec: allSentences)
    {
        std::string currentStn;
        for(auto idx: sentenceVec)
        {
            currentStn = currentStn + letters[idx].text + " ";
        }
        // std::cout<< currentStn << std::endl;
        sentence.push_back(currentStn);
    }
    printf("sentence final %d \n", sentence.size());
    return;
}

}