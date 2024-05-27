#include "RectEngine.h"

using namespace nvocdr;

RectEngine::RectEngine(const int& output_height, const int& output_width, const int& ocr_infer_batch, const bool upside_down, const bool isNHWC, const float& rot_thresh)
    : mOutputHeight(output_height)
    , mOutputWidth(output_width)
    , mOutputChannel(RECT_OUTPUT_CHANNEL)
    , mOcrInferBatch(ocr_infer_batch)
    , mUDFlag(upside_down)
    , mIsNHWC(isNHWC)
    , mRotThresh(rot_thresh)
    , mOutputBufferIndex(-1)
{
    //cublas
    checkCudaErrors(cublasCreate(&mHandle));
#ifdef RECT_DEBUG
    mImgSavePath = "/localhome/local-bizhao/dataset/pcb_images/FRAME_0_1_H.jpg";
#endif
}


RectEngine::~RectEngine()
{
    checkCudaErrors(cublasDestroy(mHandle));

}

bool RectEngine::initBuffer(BufferManager& buffer_mgr)
{
    int n = PERSPECTIVE_TRANSFORMATION_MATRIX_DIM;
    int grayBatchSize = mOcrInferBatch;
    if(mUDFlag)
    {
        grayBatchSize = mOcrInferBatch * 2;
    }
    mPtMatrixsPtrHostIdx = buffer_mgr.initHostBuffer(mOcrInferBatch, sizeof(float*));
    mInputArrayPtrHostIdx = buffer_mgr.initHostBuffer(mOcrInferBatch, sizeof(float*));
    mBarrayPtrHostIdx = buffer_mgr.initHostBuffer(mOcrInferBatch, sizeof(float*));
    mInfoArrayHostIdx = buffer_mgr.initHostBuffer(mOcrInferBatch, sizeof(int));

    mPtMatrixsPtrDevIdx = buffer_mgr.initDeviceBuffer(mOcrInferBatch, sizeof(float*));
    mPtMatrixsDevIdx = buffer_mgr.initDeviceBuffer(mOcrInferBatch*(n+1), sizeof(float));
    mPloy2ImgsDevIdx = buffer_mgr.initDeviceBuffer(mOcrInferBatch, sizeof(int));
    mPivotArrayDevIdx = buffer_mgr.initDeviceBuffer(mOcrInferBatch * n, sizeof(int));
    mInfoArrayDevIdx = buffer_mgr.initDeviceBuffer(mOcrInferBatch, sizeof(int));
    mLUArrayDevIdx = buffer_mgr.initDeviceBuffer(mOcrInferBatch*n*n, sizeof(float));
    mBarrayDevIdx = buffer_mgr.initDeviceBuffer(mOcrInferBatch*n, sizeof(float));
    mLUArrayPtrDevIdx = buffer_mgr.initDeviceBuffer(mOcrInferBatch, sizeof(float*));
    mBarrayPtrDevIdx = buffer_mgr.initDeviceBuffer(mOcrInferBatch, sizeof(float*));
    mPerspectiveOutputBufferDevIdx = buffer_mgr.initDeviceBuffer(mOcrInferBatch*3*mOutputHeight*mOutputWidth, sizeof(uchar));
    mGrayOutputBufferDevIdx = buffer_mgr.initDeviceBuffer(grayBatchSize*mOutputWidth*mOutputHeight, sizeof(float));

#ifdef RECT_DEBUG

    mPerspectiveOutputBufferHostIdx = buffer_mgr.initHostBuffer(mOcrInferBatch*3*mOutputHeight*mOutputWidth, sizeof(uchar));
    mGrayOutputBufferHostIdx = buffer_mgr.initHostBuffer(grayBatchSize*mOutputWidth*mOutputHeight, sizeof(float));
#endif
    return true;
}

bool
RectEngine::setOutputBuffer(const int& index)
{
    mOutputBufferIndex = index;
    return 0;
}

#ifdef RECT_DEBUG
int global_id = 0;
#endif
bool
RectEngine::infer(void* input_data, const Dims& input_shape, 
                  BufferManager& buffer_mgr, const std::vector<Polygon>& polys,
                  const std::vector<int>& polys_to_imgs, const cudaStream_t& stream)
{

    // Need to do transpose if isNHWC is True
    //@TODO(tylerz): do we need to re-organize the input data ???

    if(polys.size() == 0 || polys_to_imgs.size() == 0)
    {
        return false;
    }

   
    bool isNHWC = getDataFormat();

    int ptMatrixRow = PERSPECTIVE_TRANSFORMATION_MATRIX_DIM;
    int ptMatrixSize = ptMatrixRow * ptMatrixRow;
    std::vector<float> coef_matrix(ptMatrixSize * polys_to_imgs.size());
    std::vector<float> pt_Matrix_Batchs(ptMatrixRow * polys_to_imgs.size());
    
#ifdef RECT_DEBUG
    std::vector<cv::Mat> ptMats;
#endif
    for (int idx_poly=0; idx_poly<polys_to_imgs.size(); ++idx_poly)
    {
        // Format the points:
        // - Re-arrange the points to the new format:
        // - 0 left-top: left point of the top long side.
        // - 1 right-top: right point of the top long side.
        // - 2 right-bottom: right point of the bottom long side.
        // - 3 left-bottom: left point of the bottom long side.
        std::vector<Point2d> format_points(4);
        formatPoints(polys[idx_poly], format_points);
        getCoefMatrix(coef_matrix.data() + idx_poly*ptMatrixSize, pt_Matrix_Batchs.data() + idx_poly*ptMatrixRow, format_points, mOutputWidth, mOutputHeight);
#ifdef RECT_DEBUG
        cv::Point2f src_pts[4] = {cv::Point2f(format_points[0].x, format_points[0].y),
                                cv::Point2f(format_points[1].x, format_points[1].y),
                                cv::Point2f(format_points[2].x, format_points[2].y),
                                cv::Point2f(format_points[3].x, format_points[3].y)};
        cv::Point2f dst_pts[4] = {cv::Point2f(0, 0),
                                cv::Point2f(mOutputWidth, 0),
                                cv::Point2f(mOutputWidth, mOutputHeight),
                                cv::Point2f(0, mOutputHeight)};
        cv::Mat pt_mat = cv::getPerspectiveTransform(src_pts, dst_pts);
        ptMats.emplace_back(pt_mat);
#endif
    }
    // calculate multi batch perspective transformation matrix
    if(!(solveLinearEqu(coef_matrix.data(), pt_Matrix_Batchs.data(), ptMatrixRow, polys_to_imgs.size(), buffer_mgr,stream)))
    {
        std::cerr<< "[ERROR] Calculate multi batch perspective transformation matrix failed!" << std::endl;
        return false;
    }
    initPtMatrix(pt_Matrix_Batchs.data(), polys_to_imgs.size(), buffer_mgr, stream);
    // mutil batch perspective transformation
    warpPersceptive(input_data, polys_to_imgs, input_shape, mOutputWidth, mOutputHeight,  mUDFlag, isNHWC, buffer_mgr, stream);

#ifdef RECT_DEBUG
    int img_size = input_shape.d[1] * input_shape.d[2] * input_shape.d[3];
    checkCudaErrors(cudaMemcpy(buffer_mgr.mHostBuffer[mGrayOutputBufferHostIdx].data(), buffer_mgr.mDeviceBuffer[mGrayOutputBufferDevIdx].data(), buffer_mgr.mDeviceBuffer[mGrayOutputBufferDevIdx].nbBytes(), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(buffer_mgr.mHostBuffer[mPerspectiveOutputBufferHostIdx].data(), buffer_mgr.mDeviceBuffer[mPerspectiveOutputBufferDevIdx].data(), buffer_mgr.mHostBuffer[mPerspectiveOutputBufferHostIdx].size()*sizeof(uchar), cudaMemcpyDeviceToHost));
    for (int idx_poly=0; idx_poly<polys_to_imgs.size(); ++idx_poly)
    {
        // write pt image
        int pt_img_size = 3*mOutputWidth*mOutputHeight;
        
        uchar* h_pt_data = static_cast<uchar*>(buffer_mgr.mHostBuffer[mPerspectiveOutputBufferHostIdx].data() + idx_poly*pt_img_size);
        cv::Mat pt_frame(mOutputHeight, mOutputWidth, CV_8UC3, h_pt_data);
        std::string pt_img_file = mImgSavePath + std::to_string(polys_to_imgs[idx_poly]) + '_' + std::to_string(idx_poly) + "_pt_cuda.png";
        cv::imwrite(pt_img_file, pt_frame);
        
        // write  gray img
        int gray_img_size = mOutputWidth*mOutputHeight;
        float* h_gray_data = static_cast<float*>( buffer_mgr.mHostBuffer[mGrayOutputBufferHostIdx].data()+ idx_poly*gray_img_size*sizeof(float));
        uchar h_gray_data_uchar[224*224];
        float gray_data;
        for(int n=0; n<gray_img_size; ++n)
        {
             if(typeid(float) == typeid(gray_data))
            {
                h_gray_data_uchar[n] = (uchar)(h_gray_data[n]*127.5 + 127.5);
            }
            else
            {
                h_gray_data_uchar[n] = (uchar)(h_gray_data[n]);
            }
        }
        
        cv::Mat gray_frame(mOutputHeight, mOutputWidth, CV_8UC1, h_gray_data_uchar);
        std::string gray_img_file = mImgSavePath + std::to_string(polys_to_imgs[idx_poly]) + '_' + std::to_string(idx_poly) + "_gray_cuda.png";
        cv::imwrite(gray_img_file, gray_frame);
        if(mUDFlag)
        {
            // write  gray upsidedown img
            float* h_gray_UDdata = static_cast<float*>(buffer_mgr.mHostBuffer[mGrayOutputBufferHostIdx].data()+(idx_poly+polys_to_imgs.size())*gray_img_size*sizeof(float));
            for(int n=0; n<gray_img_size; ++n)
            {
                if(typeid(float) == typeid(gray_data))
                {
                    h_gray_data_uchar[n] = (uchar)(h_gray_UDdata[n]*127.5 + 127.5);
                }
                else
                {
                    h_gray_data_uchar[n] = (uchar)(h_gray_UDdata[n]);
                }
            }
            cv::Mat gray_UDframe(mOutputHeight, mOutputWidth, CV_8UC1, h_gray_data_uchar);
            std::string gray_UD_img_file = mImgSavePath + std::to_string(polys_to_imgs[idx_poly]) + '_' + std::to_string(idx_poly) + "_gray_UD_cuda.png";
            cv::imwrite(gray_UD_img_file, gray_UDframe); 
        }
    }
#endif
    return true;
}

void
RectEngine::formatPoints(const Polygon& polys, std::vector<Point2d>& format_points)
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
    float aspect_ratio = distances[0] / distances[1];
    if(aspect_ratio < mRotThresh)
    {
        Point2d tmp_point = format_points[0];
        format_points[0] = format_points[3];
        format_points[3] = format_points[2];
        format_points[2] = format_points[1];
        format_points[1] = tmp_point;
    }

    return;
}

bool
RectEngine::solveLinearEqu(float* h_AarrayInput, float* h_BarrayInput, int n, int batchSize, BufferManager& buffer_mgr, const cudaStream_t& stream)
{
    // host variables
    size_t matSize = n * n * sizeof(float);
    int lda = n;
    int h_infoBArray = -1;
    int* infoArrayHost = static_cast<int*>(buffer_mgr.mHostBuffer[mInfoArrayHostIdx].data());
    int* pivotArrayDev = static_cast<int*>(buffer_mgr.mDeviceBuffer[mPivotArrayDevIdx].data());
    int* infoArrayDev = static_cast<int*>(buffer_mgr.mDeviceBuffer[mInfoArrayDevIdx].data());
    float* luArrayDev = static_cast<float*>(buffer_mgr.mDeviceBuffer[mLUArrayDevIdx].data());
    float* barrayDev = static_cast<float*>(buffer_mgr.mDeviceBuffer[mBarrayDevIdx].data());
    float** inputArrayPtrHost = static_cast<float**>(buffer_mgr.mHostBuffer[mInputArrayPtrHostIdx].data());
    float** barrayPtrHost = static_cast<float**>(buffer_mgr.mHostBuffer[mBarrayPtrHostIdx].data());
    float** luArrayPtrDev = static_cast<float**>(buffer_mgr.mDeviceBuffer[mLUArrayPtrDevIdx].data());
    float** barrayPtrDev = static_cast<float**>(buffer_mgr.mDeviceBuffer[mBarrayPtrDevIdx].data());
    // copy data to device from host
    checkCudaErrors(cudaMemcpyAsync(luArrayDev, h_AarrayInput, batchSize * matSize, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(barrayDev, h_BarrayInput, batchSize * n * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    // create pointer array for matrices
    for (int i = 0; i < batchSize; i++)
    {
        inputArrayPtrHost[i] = luArrayDev + (i * n * n);
        barrayPtrHost[i] = barrayDev + (i * n);
    } 
    checkCudaErrors(cudaMemcpyAsync(luArrayPtrDev, inputArrayPtrHost, batchSize * sizeof(float*),cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(barrayPtrDev, barrayPtrHost, batchSize * sizeof(float*),cudaMemcpyHostToDevice, stream));

    // perform LU decomposition
    checkCudaErrors(cublasSgetrfBatched(mHandle, n, luArrayPtrDev, n, pivotArrayDev, infoArrayDev, batchSize));
    
    // copy data to host from device
    checkCudaErrors(cudaMemcpyAsync(infoArrayHost, infoArrayDev, batchSize * sizeof(int), cudaMemcpyDeviceToHost, stream));
    for (int i = 0; i < batchSize; i++) 
    {
        if (infoArrayHost[i] == 0)
        {
            continue;
        }
        else if (infoArrayHost[i] > 0) 
        {
            printf(
                "> execution for matrix %05d is successful, but U is singular and "
                "U(%d,%d) = 0..\n",
                i + 1, infoArrayHost[i] - 1, infoArrayHost[i] - 1);
        }
         else  // (infoArrayHost[i] < 0)
        {
            printf("> ERROR: matrix %05d have an illegal value at index %d = %lf..\n",
                    i + 1, -infoArrayHost[i],
                    *(h_AarrayInput + (i * n * n) + (-infoArrayHost[i])));
        }
        return false;
    }

    //solve linear equations
    checkCudaErrors(cublasSgetrsBatched(mHandle,CUBLAS_OP_N,n,1,luArrayPtrDev,lda,pivotArrayDev,barrayPtrDev,lda, &h_infoBArray,batchSize));
    checkCudaErrors(cudaMemcpyAsync(h_BarrayInput, barrayDev, batchSize * n *sizeof(int), cudaMemcpyDeviceToHost, stream));

    if (h_infoBArray != 0)
    {
        printf("> ERROR: execution is failed, %d parameter have an illegal value\n", -h_infoBArray);
        return false;
    }

    return true;
}

void
RectEngine::getCoefMatrix(float* coefMat, float* dstMat, std::vector<Point2d>& sourcePoints,int w, int h)
{
    std::vector<Point2d> destinationPoints{Point2d{0,0},Point2d{w,0},Point2d{w,h},Point2d{0,h}};
    // coef matrix is stored in row-major format
    for(int i=0; i<4; i++)
    {
        coefMat[i + 0*8] = sourcePoints[i].x;
        coefMat[i + 1*8] = sourcePoints[i].y;
        coefMat[i + 2*8] = 1;
        coefMat[i + 3*8] = 0;
        coefMat[i + 4*8] = 0;
        coefMat[i + 5*8] = 0;
        coefMat[i + 6*8] = 0 - (sourcePoints[i].x * destinationPoints[i].x);
        coefMat[i + 7*8] = 0 - (sourcePoints[i].y * destinationPoints[i].x);
        coefMat[(i+4) + 0*8] = 0;
        coefMat[(i+4) + 1*8] = 0;
        coefMat[(i+4) + 2*8] = 0;
        coefMat[(i+4) + 3*8] = sourcePoints[i].x;
        coefMat[(i+4) + 4*8] = sourcePoints[i].y;
        coefMat[(i+4) + 5*8] = 1;
        coefMat[(i+4) + 6*8] = 0 -(sourcePoints[i].x * destinationPoints[i].y);
        coefMat[(i+4) + 7*8] = 0 -(sourcePoints[i].y * destinationPoints[i].y);
        dstMat[i] = destinationPoints[i].x;
        dstMat[i+4] = destinationPoints[i].y;
    }
    return;
}

float RectEngine::det(const std::vector<float>& m)
{
    if(m.size() != 9) return -1;
    return m[0*3+0] * (m[1*3+1] * m[2*3+2] - m[1*3+2] * m[2*3+1]) + m[0*3+1] * (m[1*3+2] * m[2*3+0] - m[1*3+0] * m[2*3+2])
         + m[0*3+2] * (m[1*3+0] * m[2*3+1] - m[1*3+1] * m[2*3+0]);
}

std::vector<float> RectEngine::matrixInversion(const std::vector<float>& m)
{
    float d = det(m);
    std::vector<float>  A(9,0);
    A[0*3+0] = (m[1*3+1] * m[2*3+2] - m[1*3+2] * m[2*3+1]) / d;
    A[0*3+1] = -(m[0*3+1] * m[2*3+2] - m[0*3+2] * m[2*3+1]) / d;
    A[0*3+2] = (m[0*3+1] * m[1*3+2] - m[0*3+2] * m[1*3+1]) / d;
    A[1*3+0] = -(m[1*3+0] * m[2*3+2] - m[1*3+2] * m[2*3+0]) / d;
    A[1*3+1] = (m[0*3+0] * m[2*3+2] - m[0*3+2] * m[2*3+0]) / d;
    A[1*3+2] = -(m[0*3+0] * m[1*3+2] - m[0*3+2] * m[1*3+0]) / d;
    A[2*3+0] = (m[1*3+0] * m[2*3+1] - m[1*3+1] * m[2*3+0]) / d;
    A[2*3+1] = -(m[0*3+0] * m[2*3+1] - m[0*3+1] * m[2*3+0]) / d;
    A[2*3+2] = (m[0*3+0] * m[1*3+1] - m[0*3+1] * m[1*3+0]) / d;

    return A;
}

void 
RectEngine::initPtMatrix(const float *transMatrix, const int batchSize, BufferManager& buffer_mgr, const cudaStream_t& stream)
{

    float** ptMatrixsPtrHost = static_cast<float**>(buffer_mgr.mHostBuffer[mPtMatrixsPtrHostIdx].data());
    float** ptMatrixsPtrDev = static_cast<float**>(buffer_mgr.mDeviceBuffer[mPtMatrixsPtrDevIdx].data());
    float* matrixsDev = static_cast<float*>(buffer_mgr.mDeviceBuffer[mPtMatrixsDevIdx].data());
    int n = PERSPECTIVE_TRANSFORMATION_MATRIX_DIM;
    
    std::vector<float> ptMatInv;
    for(int i=0; i<batchSize; ++i)
    {
        std::vector<float> ptMat(transMatrix + i*n, transMatrix+ (i+1)*n);
        ptMat.emplace_back(1);
        ptMatInv = matrixInversion(ptMat);
#ifdef RECT_DEBUG
        printf("----- pt Matrix CUBLAS result -----");
        for (int i = 0; i < ptMat.size(); i++)
        {
            if(i%3==0) printf("\n");
            std::cout<< ptMat[i] << ", ";
        }
        printf("\n");
        printf("----- pt Matrix Inv CUBLAS result -----");
        for (int i = 0; i < ptMat.size(); i++)
        {
            if(i%3==0) printf("\n");
            std::cout<< ptMatInv[i] << ", ";
        }
        printf("\n");
#endif
        checkCudaErrors(cudaMemcpyAsync(matrixsDev+i*(n+1), ptMatInv.data(), (n+1)*sizeof(float), cudaMemcpyHostToDevice, stream));
        ptMatrixsPtrHost[i] = matrixsDev + i*(n+1);
    }

    checkCudaErrors(cudaMemcpyAsync(ptMatrixsPtrDev, ptMatrixsPtrHost, batchSize*sizeof(float*), cudaMemcpyHostToDevice, stream));
}

void
RectEngine::warpPersceptive(void* src,  const std::vector<int>& poly2Imgs, const Dims& input_shape, int outWeight, int outHeight, bool upsidedown, bool isNHWC, BufferManager& buffer_mgr, const cudaStream_t& stream)
{
    int inHeight = 0;
    int inWeight = 0;
    int inChannels = 0;
    int inBatchSize = input_shape.d[0];
    if (isNHWC)
    {
        inHeight = input_shape.d[1];
        inWeight = input_shape.d[2];
        inChannels = input_shape.d[3];
    }
    else
    {
        inHeight = input_shape.d[2];
        inWeight = input_shape.d[3];
        inChannels = input_shape.d[1];
    }

    int outBatchSize = poly2Imgs.size();
    uchar* src_data =  static_cast<uchar*>(src);
    uchar* dst_data =  static_cast<uchar*>(buffer_mgr.mDeviceBuffer[mPerspectiveOutputBufferDevIdx].data());
    float* dst_gray_data =  static_cast<float*>(buffer_mgr.mDeviceBuffer[mGrayOutputBufferDevIdx].data());

    ImagePtrCUDA<uchar> src_ptr(inBatchSize, inHeight, inWeight, inChannels, src_data);
    ImagePtrCUDA<uchar> dst_ptr(outBatchSize, outHeight, outWeight, inChannels, dst_data);

    int gray_batchsize = outBatchSize;
    if(upsidedown)
    {
        gray_batchsize = outBatchSize* 2;
    }
    ImagePtrCUDA<float> dst_gray_ptr(gray_batchsize, outHeight, outWeight, 1, dst_gray_data);
    int* ploy2ImgsDev = static_cast<int*>(buffer_mgr.mDeviceBuffer[mPloy2ImgsDevIdx].data());
    float** ptMatrixPtr = static_cast<float**>(buffer_mgr.mDeviceBuffer[mPtMatrixsPtrDevIdx].data());
    checkCudaErrors(cudaMemcpyAsync(ploy2ImgsDev, poly2Imgs.data(), poly2Imgs.size()*sizeof(int), cudaMemcpyHostToDevice, stream));
    warp_caller(src_ptr, dst_ptr, dst_gray_ptr, ptMatrixPtr, ploy2ImgsDev, upsidedown, stream);

}
