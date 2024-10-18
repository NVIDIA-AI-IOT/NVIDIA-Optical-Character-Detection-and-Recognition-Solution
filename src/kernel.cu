#include "kernel.h"



inline int divUp(int a, int b)
{
    assert(b > 0);
    return ceil((float)a / b);
};

template<typename Dtype>
__global__ void subscal_kernel(const unsigned int sample_cnt, void* data, float scale, float mean)
{
    Dtype* idata = static_cast<Dtype*>(data);
    unsigned int cur_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cur_id < sample_cnt)
        idata[cur_id] = (idata[cur_id] - mean) * scale;
}


void subscal(const unsigned int sample_cnt, void* data, float scale, float mean, const cudaStream_t& stream)
{
    unsigned int n_blocks = (sample_cnt + (CUDA_BLOCK_THREADS - 1)) / CUDA_BLOCK_THREADS;
    subscal_kernel<float><<<n_blocks, CUDA_BLOCK_THREADS, 0, stream>>>(
        sample_cnt, data, scale, mean
    );
}

__device__ __forceinline__ float2 calcCoord(const float *c_warpMat, int x, int y)
{
    const float coeff = 1.0f / (c_warpMat[6] * x + c_warpMat[7] * y + c_warpMat[8]);
    const float xcoo = coeff * (c_warpMat[0] * x + c_warpMat[1] * y + c_warpMat[2]);
    const float ycoo = coeff * (c_warpMat[3] * x + c_warpMat[4] * y + c_warpMat[5]);

    return make_float2(xcoo, ycoo);
}

template<typename T>
__global__ void warp(ImagePtrCUDA<T> srcPtr, ImagePtrCUDA<float> dst, ImagePtrCUDA<float> dst_gray, float** ptMatrixPtr, int* ploy2Imgs, bool upsidedown, bool onlyRGB=false)
{
    
    const int  dst_b = blockIdx.z;
    const int src_b = ploy2Imgs[dst_b];
    const int  x     = blockDim.x * blockIdx.x + threadIdx.x;
    const int  y     = blockDim.y * blockIdx.y + threadIdx.y;
    const int  c     = threadIdx.z;
    if (x < dst.cols && y < dst.rows)
    {
        const float2 coord = calcCoord(ptMatrixPtr[dst_b], x, y);
        *dst.ptr(dst_b, y, x, c) = linearInterp<T, float>(srcPtr, src_b, coord.y, coord.x, c);
        __syncthreads();
        
        int b   = *dst.ptr(dst_b, y, x, 0);
        int g   = *dst.ptr(dst_b, y, x, 1);
        int r   = *dst.ptr(dst_b, y, x, 2);
        if (onlyRGB)
        {
            // BGR to RGB and norm
            // mean value from https://github.com/mlfoundations/open_clip/blob/b4cf9269b0b11c0eea47cb16039369a46bd67449/src/open_clip/constants.py
            *dst.ptr(dst_b, y, x, 0) = (r/255.0 - 0.4814546)/0.26862954;
            *dst.ptr(dst_b, y, x, 1) = (g/255.0 - 0.4578275)/0.26130258;
            *dst.ptr(dst_b, y, x, 2) = (b/255.0 - 0.4082107)/0.27577711;
            return ;
        }
        // bgr to gray
        T gray  = (T)CV_DESCALE(b * BY15 + g * GY15 + r * RY15, GRAY_SHIFT);
        float gray_scale   = ((float)gray - IMG_MEAN_GRAY) * IMG_SCALE_GRAY;
        *dst_gray.ptr(dst_b, y, x, 0) = gray_scale;
        if(upsidedown)
        {
            int dst_b_UD = dst_b + dst.batches;
            int x_UD = (dst.cols - 1 - x);
            int y_UD = (dst.rows - 1 - y);
            *dst_gray.ptr(dst_b_UD, y_UD, x_UD, 0) = gray_scale;
        }
    }
}

template<typename T>
struct WarpDispatcher
{
    static void call(const ImagePtrCUDA<T> src, ImagePtrCUDA<float> dst, ImagePtrCUDA<float> dst_gray, float** ptMatrixPtr, int* ploy2Imgs,  bool upsidedown, bool onlyRGB,
                     const cudaStream_t& stream)
    {

        dim3 block(BLOCK, BLOCK / 4, 3);
        dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y), dst.batches);

        // share memory  to save persceptive trans matrix ptr
        size_t smem_size = dst.batches * sizeof(float*);
        warp<<<grid, block, smem_size, stream>>>(src, dst, dst_gray, ptMatrixPtr, ploy2Imgs, upsidedown, onlyRGB);
        cudaStreamSynchronize(stream);
        checkKernelErrors();
    }
};

void warp_caller(const ImagePtrCUDA<uchar>& src, ImagePtrCUDA<float>& dst, ImagePtrCUDA<float>& dst_gray, float** ptMatrixPtr, int* poly2Imgs, bool upsidedown, bool isRGBInput, const cudaStream_t& stream)
{
    WarpDispatcher<uchar>::call(src, dst, dst_gray, ptMatrixPtr, poly2Imgs, upsidedown, isRGBInput, stream);
}



__global__ void nhwc2nchwWithMeanScaleResize(ImagePtrCUDA<uchar> src , ImagePtrCUDA<float> dst)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int dst_b     = blockIdx.z;
    const int dst_c     = threadIdx.z;
    const float scale_x = ((float)src.cols) / dst.cols;
    const float scale_y = ((float)src.rows) / dst.rows;

    if (dst_x < dst.cols && dst_y < dst.rows)
    {
        float fy = (float)((dst_y + 0.5f) * scale_y - 0.5f);
        float fx = (float)((dst_x + 0.5f) * scale_x - 0.5f);
        *dst.ptr(dst_b, dst_y, dst_x, dst_c) = linearInterp<uchar, float>(src, dst_b, fy, fx, dst_c);
       
        if(dst_c == 0)
        {
            *dst.ptr(dst_b, dst_y, dst_x, dst_c) -= IMG_MEAN_B;
        }
        else if(dst_c == 1)
        {
            *dst.ptr(dst_b, dst_y, dst_x, dst_c) -= IMG_MEAN_G;
        }
        else if (dst_c == 2)
        {
            *dst.ptr(dst_b, dst_y, dst_x, dst_c) -= IMG_MEAN_R;
        }
        *dst.ptr(dst_b, dst_y, dst_x, dst_c) *= IMG_SCALE_BRG;
    }
    return;
}


__global__ void threshold(ImagePtrCUDA<float> src , ImagePtrCUDA<uchar> dst, const float binaryThreshold)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int dst_b     = blockIdx.z;
    const int dst_c     = threadIdx.z;

    if (dst_x < dst.cols && dst_y < dst.rows)
    {
        if(*src.ptr(dst_b, dst_y, dst_x, dst_c) > binaryThreshold)
        {
            *dst.ptr(dst_b, dst_y, dst_x, dst_c) =   255;
        }
        else
        {
            *dst.ptr(dst_b, dst_y, dst_x, dst_c) =   0;
        }
    }

}

void blobFromImagesCUDA(void* inData, void* outData, const nvinfer1::Dims& inShape, const nvinfer1::Dims& outShape, bool inputIsNHWC, const cudaStream_t& stream)
{
    int outBatchSize = outShape.d[0];
    int outChannels = outShape.d[1];
    int outHeight = outShape.d[2];
    int outWidth = outShape.d[3];

    int inBatchSize = inShape.d[0];
    int inHeight = 0;
    int inWeight = 0;
    int inChannels = 0;
    if (inputIsNHWC)
    {
        inHeight = inShape.d[1];
        inWeight = inShape.d[2];
        inChannels = inShape.d[3];
    }
    else
    {
        inHeight = inShape.d[2];
        inWeight = inShape.d[3];
        inChannels = inShape.d[1];
    }
    ImagePtrCUDA<uchar> srcPtr(inBatchSize, inHeight, inWeight, inChannels, static_cast<uchar*>(inData), inputIsNHWC);
    ImagePtrCUDA<float> dstPtr(outBatchSize, outHeight, outWidth, outChannels, static_cast<float*>(outData), false);
    dim3 block(BLOCK, BLOCK / 4, 3);
    dim3 grid(divUp(dstPtr.cols, block.x), divUp(dstPtr.rows, block.y), dstPtr.batches);

    nhwc2nchwWithMeanScaleResize<<<grid, block, 0, stream>>>(srcPtr, dstPtr);
    cudaStreamSynchronize(stream);
    checkKernelErrors();
}


void thresholdCUDA(void* inData, void* outData, const nvinfer1::Dims& inShape, const float binaryThreshold, const cudaStream_t& stream)
{
    int inBatchSize = inShape.d[0];
    int inChannels = inShape.d[1];
    int inHeight = inShape.d[2];
    int inWeight = inShape.d[3];
    ImagePtrCUDA<float> srcPtr(inBatchSize, inHeight, inWeight, inChannels, static_cast<float*>(inData), false);
    ImagePtrCUDA<uchar> dstPtr(inBatchSize, inHeight, inWeight, inChannels, static_cast<uchar*>(outData), false);
    dim3 block(BLOCK, BLOCK / 4, 1);
    dim3 grid(divUp(srcPtr.cols, block.x), divUp(srcPtr.rows, block.y), srcPtr.batches);
    threshold<<<grid, block, 0, stream>>>(srcPtr, dstPtr, binaryThreshold);

    cudaStreamSynchronize(stream);
    checkKernelErrors();
}

__global__ void calculateRotateCoef(float *aCoeffs, const int degrees)
{
    int angle = blockIdx.x * blockDim.x + threadIdx.x;
    if(angle < degrees)
    {
        aCoeffs[2 * angle]      = cos(angle * PI / 180);
        aCoeffs[2 * angle + 1]  = sin(angle * PI / 180);
    }
}

void calculateRotateCoefCUDA(void* rotateCoefBuf, const int degrees,  const cudaStream_t& stream)
{
    dim3 block(BLOCK * 8);
    dim3 grid(divUp(degrees, block.x));
    float* aCoeffs = static_cast<float*>(rotateCoefBuf);
    calculateRotateCoef<<<grid, block, 0, stream>>>(aCoeffs, degrees);
    
}

__global__ void calculateRotateArea(ContourPtrCUDA<uint16_t> inContourPointsData, MinAreaRectPtrCUDA<int> outMinAreaRectBox, float* rotateCoeffs, int* numPointsInContourBuf, int* contoursToImagesBuf)
{
   
    int pointIdx    = blockIdx.x * blockDim.x + threadIdx.x;
    int contourIdx  = blockIdx.y;
    
    int angleIdx       = blockIdx.z;
    extern  __shared__ float rotateCoeffs_sm[];
    rotateCoeffs_sm[2 * angleIdx] = rotateCoeffs[2 * angleIdx];
    rotateCoeffs_sm[2 * angleIdx + 1] = rotateCoeffs[2 * angleIdx + 1];
    __syncthreads();
    
    if(pointIdx < numPointsInContourBuf[contourIdx])
    {
        float px = *inContourPointsData.ptr(contourIdx, pointIdx, 0);
        float py = *inContourPointsData.ptr(contourIdx, pointIdx, 1);
        float cos_coeff = rotateCoeffs_sm[2 * angleIdx];
        float sin_coeff = rotateCoeffs_sm[2 * angleIdx + 1];
        int px_rot = (px * cos_coeff) - (py * sin_coeff);
        int py_rot = (px * sin_coeff) + (py * cos_coeff);
        //xmin
        atomicMin(outMinAreaRectBox.pointPtr(contourIdx,angleIdx,0),px_rot);
        //ymin
        atomicMin(outMinAreaRectBox.pointPtr(contourIdx,angleIdx,1),py_rot);
        //xmax
        atomicMax(outMinAreaRectBox.pointPtr(contourIdx,angleIdx,2),px_rot);
        //ymax
        atomicMax(outMinAreaRectBox.pointPtr(contourIdx,angleIdx,3),py_rot);
        // TODO(@binz) can this func sync across all global block?
        __threadfence();
        int rectWidth = *outMinAreaRectBox.pointPtr(contourIdx,angleIdx,2) - *outMinAreaRectBox.pointPtr(contourIdx,angleIdx,0);
        int rectHeight = *outMinAreaRectBox.pointPtr(contourIdx,angleIdx,3) - *outMinAreaRectBox.pointPtr(contourIdx,angleIdx,1);
        *outMinAreaRectBox.pointPtr(contourIdx,angleIdx,4) = rectWidth * rectHeight;
        *outMinAreaRectBox.pointPtr(contourIdx,angleIdx,5) = angleIdx;
    }
}

__global__ void findMinAreaAndAngle(MinAreaRectPtrCUDA<int> outMinAreaRectBox, const int numOfDegrees)
{
    int angleIdx = threadIdx.x;
	if (angleIdx > numOfDegrees)
    {
        return;
    }
		
    int rectIdx = blockIdx.x;
    extern __shared__ int areaAngleBuf_sm[];
    areaAngleBuf_sm[2*angleIdx] = *outMinAreaRectBox.pointPtr(rectIdx, angleIdx, 4);
    areaAngleBuf_sm[2*angleIdx + 1] = *outMinAreaRectBox.pointPtr(rectIdx, angleIdx, 5);
    __syncthreads();
   
    for (int stride = numOfDegrees/2; stride >0; stride >>=1)
	{
		if (angleIdx < stride)
		{
            int* curAreaIdx     = &areaAngleBuf_sm[2*angleIdx];
            int* nextAreaIdx    = &areaAngleBuf_sm[2*(angleIdx + stride)];
            int* curAngleIdx    = &areaAngleBuf_sm[2*angleIdx + 1];
            int* nextAngleIdx   = &areaAngleBuf_sm[2*(angleIdx + stride) + 1];
            if(*curAreaIdx > *nextAreaIdx)
            {
                *curAreaIdx = *nextAreaIdx;
                *curAngleIdx = *nextAngleIdx;
            }	    
		}
		__syncthreads();

        if(stride%2 == 1 && areaAngleBuf_sm[0] > areaAngleBuf_sm[2*(stride - 1)])
        {
            areaAngleBuf_sm[0]  = areaAngleBuf_sm[2*(stride - 1) ];
            areaAngleBuf_sm[1]  = areaAngleBuf_sm[2*(stride - 1) + 1];
        }
        __syncthreads();
	}
    if(numOfDegrees%2 == 1 && areaAngleBuf_sm[0] > areaAngleBuf_sm[2*(numOfDegrees - 1)])
    {
        areaAngleBuf_sm[0]  = areaAngleBuf_sm[2*(numOfDegrees - 1) ];
        areaAngleBuf_sm[1]  = areaAngleBuf_sm[2*(numOfDegrees - 1) + 1];
    }

    int minRotateAngle = areaAngleBuf_sm[1];
    float cos_coeff  = cos(- minRotateAngle * PI / 180);
    float sin_coeff  = sin(- minRotateAngle * PI / 180);
    float xmin = *outMinAreaRectBox.pointPtr(rectIdx, areaAngleBuf_sm[1], 0);
    float ymin = *outMinAreaRectBox.pointPtr(rectIdx, areaAngleBuf_sm[1], 1);
    float xmax = *outMinAreaRectBox.pointPtr(rectIdx, areaAngleBuf_sm[1], 2);
    float ymax = *outMinAreaRectBox.pointPtr(rectIdx, areaAngleBuf_sm[1], 3);
    float tl_x = (xmin * cos_coeff) - (ymin * sin_coeff);
    float tl_y = (xmin * sin_coeff) + (ymin * cos_coeff);
    float br_x = (xmax * cos_coeff) - (ymax * sin_coeff);
    float br_y = (xmax * sin_coeff) + (ymax * cos_coeff);
    float tr_x = (xmax * cos_coeff) - (ymin * sin_coeff);
    float tr_y = (xmax * sin_coeff) + (ymin * cos_coeff);
    float bl_x = (xmin * cos_coeff) - (ymax * sin_coeff);
    float bl_y = (xmin * sin_coeff) + (ymax * cos_coeff);
    
    *outMinAreaRectBox.minAreaPointPtr(rectIdx, 0, 0) = bl_x;
    *outMinAreaRectBox.minAreaPointPtr(rectIdx, 0, 1) = bl_y;
    *outMinAreaRectBox.minAreaPointPtr(rectIdx, 1, 0) = tl_x;
    *outMinAreaRectBox.minAreaPointPtr(rectIdx, 1, 1) = tl_y;
    *outMinAreaRectBox.minAreaPointPtr(rectIdx, 2, 0) = tr_x;
    *outMinAreaRectBox.minAreaPointPtr(rectIdx, 2, 1) = tr_y;
    *outMinAreaRectBox.minAreaPointPtr(rectIdx, 3, 0) = br_x;
    *outMinAreaRectBox.minAreaPointPtr(rectIdx, 3, 1) = br_y;

}
void minAreaRectCUDA(ContourPtrCUDA<uint16_t>& inContourPointsData, MinAreaRectPtrCUDA<int>& outMinAreaRectBox, void* rotateCoeffs, void* numPointsInContourBuf, void* contoursToImages,  const int numContours, const int maxNumPointsInContour, const int numOfDegrees, const cudaStream_t& stream)
{
    
    dim3 block(BLOCK * 8);
    dim3 grid(divUp(maxNumPointsInContour, block.x), numContours, numOfDegrees);
    // sm for rotate coeff
    size_t smem_size = 2 * numOfDegrees * sizeof(float);
    calculateRotateArea<<<grid, block, smem_size, stream>>>(inContourPointsData, outMinAreaRectBox, static_cast<float*>(rotateCoeffs), static_cast<int*>(numPointsInContourBuf), static_cast<int*>(contoursToImages));
    cudaStreamSynchronize(stream);
    checkKernelErrors();
    dim3 grid1(numContours);
  
    findMinAreaAndAngle<<<grid1, block, smem_size, stream>>>(outMinAreaRectBox, numOfDegrees);
    cudaStreamSynchronize(stream);
    checkKernelErrors();
}


__global__ void mergePatchKernel(ImagePtrCUDA<uchar> patchThreshold , ImagePtrCUDA<uchar> mergeThreshold, ImagePtrCUDA<float> patchRaw , ImagePtrCUDA<float> mergeRaw,  const ImgROI patchROI, const ImgROI mergeROI)
{
    int patch_x     = blockIdx.x * blockDim.x + threadIdx.x;
    int patch_y     = blockIdx.y * blockDim.y + threadIdx.y;

    if(patch_x < patchROI.w && patch_y < patchROI.h)
    {
        *mergeThreshold.ptr(0, mergeROI.y + patch_y, mergeROI.x + patch_x, 0) = *patchThreshold.ptr(0,patch_y+ patchROI.y,patch_x+patchROI.x, 0);
        *mergeRaw.ptr(0, mergeROI.y + patch_y, mergeROI.x + patch_x, 0) = *patchRaw.ptr(0,patch_y+ patchROI.y,patch_x+patchROI.x, 0);
    }
    return;

}


void patchMergeWarp(void* patchThresholdData, void* mergeThresholdData, void* patchOcdOutRawData, void* mergeOcdOutRawData, const nvinfer1::Dims& patchShape, const nvinfer1::Dims& mergeShape, const ImgROI& patchROI, const ImgROI& mergeROI, const cudaStream_t& stream)
{   
    int patchBS = patchShape.d[0];
    int patchCh = patchShape.d[1];
    int patch_h = patchShape.d[2];
    int patch_w = patchShape.d[3];

    int mergeBS = mergeShape.d[0];
    int mergeCh = mergeShape.d[3];
    int merge_h = mergeShape.d[1];
    int merge_w = mergeShape.d[2];

    ImagePtrCUDA<uchar> patchThresholdPtr(patchBS, patch_h, patch_w, patchCh, static_cast<uchar*>(patchThresholdData), false);
    ImagePtrCUDA<float> patchRawPtr(patchBS, patch_h, patch_w, patchCh, static_cast<float*>(patchOcdOutRawData), false);
    ImagePtrCUDA<uchar> mergeThresholdPtr(mergeBS, merge_h, merge_w, mergeCh, static_cast<uchar*>(mergeThresholdData), true);
    ImagePtrCUDA<float> mergeRawPtr(mergeBS, merge_h, merge_w, mergeCh, static_cast<float*>(mergeOcdOutRawData), true);

    dim3 block(BLOCK,BLOCK);
    dim3 grid(divUp(patchROI.w, block.x),divUp(patchROI.h, block.y));

    mergePatchKernel<<<grid, block, 0 , stream>>>(patchThresholdPtr, mergeThresholdPtr, patchRawPtr, mergeRawPtr,  patchROI, mergeROI);
    cudaStreamSynchronize(stream);
    checkKernelErrors();
}

//HWC as input
template<typename Dtype>
__device__ Dtype value_at_row_col_channel(Dtype *im, int rowIdx, int colIdx, int chIdx, int batchIdx, int colStride, int rowStride, int batchStride)
{
    Dtype *p = im + chIdx + ((int)colIdx) * colStride + ((int)rowIdx) * rowStride + ((int)batchIdx)*batchStride;
    return p[0];
}

template<typename Dtype>
__device__ void set_value_at_row_col_channel(Dtype vP_ch, Dtype *out_im, int rowIdx, int colIdx, int chIdx, int batchIdx, int colStride, int rowStride, int batchStride)
{
    Dtype *p = out_im + chIdx + ((int)colIdx) * colStride + ((int)rowIdx) * rowStride + ((int)batchIdx)*batchStride;
    p[0] = vP_ch;
}

template<typename Dtype>
__host__ __device__ __forceinline__  Dtype bordConstant(Dtype* src, int src_w, int src_h, int rowIdx, int colIdx, int chIdx, int batchIdx, int colStride, int rowStride, int batchStride)
{
    if((float)colIdx >= 0 && colIdx < src_w && (float)rowIdx >= 0 && rowIdx < src_h)
    {
        return value_at_row_col_channel(src, rowIdx, colIdx,  chIdx, batchIdx, colStride, rowStride, batchStride);
    }
    else 
    {
        return (Dtype)-1;
    }
}

template<typename Dtype>
__global__ void resize(
    Dtype *im, 
    int im_hh,
    int im_ww,
    int out_h,
    int out_w, 
    int channels, 
    int batchSize, 
    int dstColStride, 
    int dstRowStride, 
    int dstBatchStride, 
    float scale_w, 
    float scale_h, 
    Dtype *out_im
    )
{

    // float scale_w = float(im_ww)/float(out_w);
    // float scale_h = float(im_hh)/float(out_h);
    int srcColStride = channels;
    int srcRowStride = channels * im_ww;
    int srcBatchStride = channels * im_ww * im_hh;



    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = threadIdx.z;
    int b_out = blockIdx.z;
    
    if (x_out < out_w && y_out < out_h)
    {

        float x_out_float = (float) x_out;
        float y_out_float = (float) y_out;
        float out = 0;
        
        //remapped to source image, called P
        float P_x_in_float = scale_w * (x_out_float+ 0.5) - 0.5;
        float P_y_in_float = scale_h * (y_out_float+ 0.5) - 0.5;
        
        float x1 = floorf(P_x_in_float);
        float y1 = floorf(P_y_in_float);
        
        float x2 = x1 + 1;
        float y2 = y1 + 1;
        
        Dtype src_reg = bordConstant(im, im_ww, im_hh, y1, x1, c_out, b_out, srcColStride, srcRowStride, srcBatchStride );
        out = out + src_reg * ((x2 - P_x_in_float) * (y2 - P_y_in_float));

        src_reg = bordConstant(im, im_ww, im_hh, y1, x2, c_out, b_out, srcColStride, srcRowStride, srcBatchStride );
        out = out + src_reg * ((P_x_in_float - x1) * (y2 - P_y_in_float));

        src_reg = bordConstant(im, im_ww, im_hh, y2, x1, c_out, b_out, srcColStride, srcRowStride, srcBatchStride );
        out = out + src_reg * ((x2 - P_x_in_float) * (P_y_in_float - y1));

        src_reg = bordConstant(im, im_ww, im_hh, y2, x2, c_out, b_out, srcColStride, srcRowStride, srcBatchStride );
        out = out + src_reg * ((P_x_in_float - x1) * (P_y_in_float - y1));

        
        set_value_at_row_col_channel((Dtype)out, out_im, y_out, x_out, c_out, b_out, dstColStride, dstRowStride, dstBatchStride);

    }
}

float
KeepAspectRatioResize(void* inData, void* outData, const nvinfer1::Dims& inShape,
                      const int32_t out_h, const int32_t out_w, const cudaStream_t& stream)
{
    //Only support NHWC now
    int32_t batch_size = inShape.d[0];
    int32_t img_h = inShape.d[1];
    int32_t img_w = inShape.d[2];
    int32_t img_c = inShape.d[3];

    float ar_inp = float(img_w)/float(img_h);
    float ar_out = float(out_w)/float(out_h);

    int32_t new_width = 0;
    int32_t new_height = 0;
    float scale_w = 1.0f;
    float scale_h = 1.0f;
    float rescale = 1.0f;

    if (ar_inp >= ar_out)
    {
        new_width = out_w;
        new_height = new_width / ar_inp;
        rescale = float(img_w) / new_width;
    }
    else
    {
        new_height = out_h;
        new_width = new_height * ar_inp;
        rescale = float(img_h) / new_height;
    }

    scale_w = float(img_w) / float(new_width);
    scale_h = float(img_h) / float(new_height);

    int32_t dst_col_stride = img_c;
    int32_t dst_row_stride = img_c * out_w;
    int32_t dst_batch_stride = img_c * out_w * out_h;

    int32_t threads_num = 16;
    dim3 block(threads_num, threads_num, img_c);
    dim3 grid(divUp(new_width, threads_num), divUp(new_height, threads_num), batch_size);
    
    resize<uchar><<<grid, block, 0>>>(reinterpret_cast<uchar*>(inData), img_h, img_w, new_height, new_width, img_c,
                                               batch_size, dst_col_stride, dst_row_stride, dst_batch_stride,
                                               scale_w, scale_h, reinterpret_cast<uchar*>(outData));
    // cudaStreamSynchronize(stream);
    // checkKernelErrors();
    return rescale;
}

__global__ void calculateBoxDistanceLauncher(float* boxCenters,float* boxLeftCenters ,float* boxRightCenters, short* letterMask, float* r2lDistance, const int letterNum)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    if(j < letterNum)
    {
        float box1_x =  boxCenters[2*i];
        float box1_y =  boxCenters[2*i + 1];
        float box2_x =  boxCenters[2*j];
        float box2_y =  boxCenters[2*j + 1];
        double slope = -(box2_y - box1_y) / (box2_x - box1_x + EXP);
        float radian = atan(slope);
        float angle = abs(radian/PI *180);

        // box1 left/right side center
        float box1_lx = boxLeftCenters[2*i];
        float box1_ly = boxLeftCenters[2*i + 1];
        float box1_rx = boxRightCenters[2*i];
        float box1_ry = boxRightCenters[2*i + 1];
        // box2 left/right side center
        float box2_lx = boxLeftCenters[2*j];
        float box2_ly = boxLeftCenters[2*j + 1];
        float box2_rx = boxRightCenters[2*j];
        float box2_ry = boxRightCenters[2*j + 1];

        float left2RightDis = sqrt( (box1_lx -box2_rx)*(box1_lx -box2_rx) + (box1_ly -box2_ry)*(box1_ly -box2_ry));
        float right2LeftDis = sqrt( (box1_rx -box2_lx)*(box1_rx -box2_lx) + (box1_ry -box2_ly)*(box1_ry -box2_ly));
        r2lDistance[i*MAX_LETTERS_IN_IMAGE + j] = right2LeftDis;
        if(angle < ANGLE_THRESHOLD )
        {
            letterMask[i*MAX_LETTERS_IN_IMAGE + j] = 1;
        }
        else
        {
            letterMask[i*MAX_LETTERS_IN_IMAGE + j] = -1;
        }
    }
    

}


void calculateBoxDistance(void* boxCenters, void* boxLeftCenters, void* boxRightCenters, void* letterMask, void* r2lDistance, const int letterNum, const cudaStream_t& stream)
{

    dim3 block(BLOCK * divUp(letterNum, BLOCK));
    dim3 grid(letterNum);

    float* boxCentersMetric = static_cast<float*>(boxCenters);
    float* boxLeftCentersMetric = static_cast<float*>(boxLeftCenters);
    float* boxRightCentersMetric = static_cast<float*>(boxRightCenters);
    short* letterMaskMetric = static_cast<short*>(letterMask);
    float* r2lDistanceMetric = static_cast<float*>(r2lDistance);
    calculateBoxDistanceLauncher<<<grid, block,0, stream>>>(boxCentersMetric, boxLeftCentersMetric, boxRightCentersMetric, letterMaskMetric,r2lDistanceMetric, letterNum);
    // // sm for rotate coeff
    // size_t smem_size = 2 * numOfDegrees * sizeof(float);
    // calculateRotateArea<<<grid, block, smem_size, stream>>>(inContourPointsData, outMinAreaRectBox, static_cast<float*>(rotateCoeffs), static_cast<int*>(numPointsInContourBuf), static_cast<int*>(contoursToImages));
    // cudaStreamSynchronize(stream);
    // checkKernelErrors();
    // dim3 grid1(numContours);
  
    // findMinAreaAndAngle<<<grid1, block, smem_size, stream>>>(outMinAreaRectBox, numOfDegrees);
    cudaStreamSynchronize(stream);
    checkKernelErrors();
}

