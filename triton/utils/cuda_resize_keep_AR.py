import os
import cv2
import time
import pycuda
from nvjpeg import NvJpeg
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

def div_up(m, n):
    """Division with round to up."""
    return int(m // n + (m % n > 0))

def allocate_cuda_array_mem_and_copy(array):
    array = array.astype(np.float32)
    array_g = cuda.mem_alloc(array.nbytes)
    cuda.memcpy_htod(array_g, array.reshape([-1]))
    return array_g

def allocate_cuda_int_mem_and_copy(int):
    h_np = np.zeros(1).astype(np.int32)
    h_np[0]=int
    q_shape_np_gpu = cuda.mem_alloc(h_np.nbytes)
    cuda.memcpy_htod(q_shape_np_gpu, h_np)
    return q_shape_np_gpu, h_np
    

mod = SourceModule("""

  //HWC as input
    __device__ float value_at_row_col_channel(float *im, int rowIdx, int colIdx, int chIdx, int batchIdx, int colStride, int rowStride, int batchStride)
    {
        float *p = im + chIdx + ((int)colIdx) * colStride + ((int)rowIdx) * rowStride + ((int)batchIdx)*batchStride;
        return p[0];
    }
  
    __device__ void set_value_at_row_col_channel(float vP_ch, float *out_im, int rowIdx, int colIdx, int chIdx, int batchIdx, int colStride, int rowStride, int batchStride)
    {
        float *p = out_im + chIdx + ((int)colIdx) * colStride + ((int)rowIdx) * rowStride + ((int)batchIdx)*batchStride;
        p[0] = vP_ch;
    }

    __host__ __device__ __forceinline__  float bordConstant(float* src, int src_w, int src_h, int rowIdx, int colIdx, int chIdx, int batchIdx, int colStride, int rowStride, int batchStride)
    {
        if((float)colIdx >= 0 && colIdx < src_w && (float)rowIdx >= 0 && rowIdx < src_h)
        {
            return value_at_row_col_channel(src, rowIdx, colIdx,  chIdx, batchIdx, colStride, rowStride, batchStride);
        }
        else 
        {
            return -1.0;
        }
    }

    __global__ void resize(
        float *im, 
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
        float *out_im
        
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
            
            float x2 = ceilf(P_x_in_float);
            float y2 = ceilf(P_y_in_float);
            
            float src_reg = bordConstant(im, im_ww, im_hh, y1, x1, c_out, b_out, srcColStride, srcRowStride, srcBatchStride );
            out = out + src_reg * ((x2 - P_x_in_float) * (y2 - P_y_in_float));

            src_reg = bordConstant(im, im_ww, im_hh, y1, x2, c_out, b_out, srcColStride, srcRowStride, srcBatchStride );
            out = out + src_reg * ((P_x_in_float - x1) * (y2 - P_y_in_float));

            src_reg = bordConstant(im, im_ww, im_hh, y2, x1, c_out, b_out, srcColStride, srcRowStride, srcBatchStride );
            out = out + src_reg * ((x2 - P_x_in_float) * (P_y_in_float - y1));

            src_reg = bordConstant(im, im_ww, im_hh, y2, x2, c_out, b_out, srcColStride, srcRowStride, srcBatchStride );
            out = out + src_reg * ((P_x_in_float - x1) * (P_y_in_float - y1));

           
            
            set_value_at_row_col_channel(out, out_im, y_out, x_out, c_out, b_out, dstColStride, dstRowStride, dstBatchStride);

        }
    }

""")

def image_resize(images, out_w, out_h, keep_ar):
    # print(f'cuda imgs shape {images.shape}, nbytes {images.nbytes}')
    image_g = allocate_cuda_array_mem_and_copy(images)
    batch = images.shape[0]
    image_h = images.shape[1]
    image_w = images.shape[2]
    image_c = images.shape[3]

    out_np = np.zeros(batch*out_h*out_w*3).astype(np.float32)
    out_np_g = allocate_cuda_array_mem_and_copy(out_np)

    if keep_ar:
        aspect_ratio_input_image = image_w/image_h
        aspect_ratio_output_image = out_w/out_h

        if (aspect_ratio_input_image >= aspect_ratio_output_image):

            new_width = int(out_w)
            new_height =  int(new_width / aspect_ratio_input_image)
            scale_w = image_w/new_width
            scale_h = image_w/new_width
        else:

            new_height = int(out_h)
            new_width = int(aspect_ratio_input_image * new_height)
            scale_w = image_h/new_height
            scale_h = image_h/new_height
    else:

        new_height = out_h
        new_width = out_w
        scale_w = image_w/new_width
        scale_h = image_h/new_height

    dstColStride = image_c
    dstRowStride = image_c * out_w
    dstBatchStride = image_c * out_w * out_h

    THREADS = 16
    blocks = (THREADS, THREADS , image_c)
    grids = ( div_up(new_width, THREADS), div_up(new_height, THREADS), batch)

    func = mod.get_function("resize")
    func(image_g, 
        np.int32(image_h),
        np.int32(image_w),
        np.int32(new_height),
        np.int32(new_width),
        np.int32(image_c),
        np.int32(batch),
        np.int32(dstColStride),
        np.int32(dstRowStride),
        np.int32(dstBatchStride),
        np.float32(scale_w),
        np.float32(scale_h),
        out_np_g, 
        grid= grids,
        block= blocks
        )

    cuda.memcpy_dtoh(out_np, out_np_g)
    out_np = out_np.reshape((-1, out_h, out_w, 3)).astype(np.uint8)
    image_g.free()
    out_np_g.free()
    
    return out_np