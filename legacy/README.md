# nvOCDR

nvOCDR is a C++ library for optical character detection and recognition. It is optimized for Nvidia devices with Nvidia software stack. 
This library consumes the TAO Toolkit trained OCDNet and OCRNet models for any OCR application.
Whether you are building a surveillance system, a traffic monitoring application, or any other type of video analytics solution, the nvOCDR library is an essential tool for achieving accurate and reliable results. It can be easily integrated to any application requiring OCR ability.

c++_samples/sample_inference --ocd_model ../onnx_models/dcn_resnet18.engine  --ocr_model ../onnx_models/ocrnet_resnet50.engine --max_candidates 100 --image ../c++_samples/test_img/scene_text.jpg --text_polygon_thresh 0.2 --show_score 1 --debug_log 0 --strategy smart --rec_all_direction 1 --ocd_type normal --debug_image 0 --binary_lower_bound 0.0 --binary_upper_bound 0.1

## Table of Contents:
- [Installation](#installation)
- [Usage](#usage)
    - [Use it in your application](#use-it-in-your-application)
    - [Use it in DeepStream](#use-nvocdr-in-deepstream-sdk)
    - [API Reference](#api-reference)
- [License](#license)

## Installation
### Prerequisites
- CUDA 11.4 or above
- TensorRT 8.5 or above (To use ViT-based model, TensorRT 8.6 above is required.)
- OpenCV 4.0 or above
- Jetpack 5.1 or above on Jetson devices
- Pretrained OCDNet and OCRNet model

#### **Set up the development environment**:
We suggest to start from TensorRT container:

- On X86 platform:
    ```shell
    docker run --gpus=all -v <work_path>:<work_path> --rm -it --privileged --net=host nvcr.io/nvidia/tensorrt:23.11-py3 bash
    # install opencv
    apt update && apt install -y libopencv-dev
    ```
- On Jetson platform
    ```shell
    docker run --gpus=all -v <work_path>:<work_path> --rm -it --privileged --net=host nvcr.io/nvidia/l4t-tensorrt:r8.5.2.2-devel bash
    # install opencv
    apt update && apt install -y libopencv-dev
    ```

#### **Prepare the OCDNet and OCRNet model**:
And then you could dowload the pretrained models of [OCDNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/ocdnet) and [OCRNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/ocrnet) with following instructions or train your own model (Please ref to TAO Toolkit documentation for how to train your own [OCDNet](https://docs.nvidia.com/tao/tao-toolkit/text/object_detection/ocd.html) and [OCRNet](https://docs.nvidia.com/tao/tao-toolkit/text/character_recognition/ocrnet.html). And there will be a vocabulary list named `character_list.txt` of OCRNet model when you download the PTM from NGC. 

- download the onnx models of OCDnet and OCRnet
```shell
mkdir onnx_models
cd onnx_models

# Download OCDnet onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocdnet/deployable_v1.0/files?redirect=true&path=dcn_resnet18.onnx' -O dcn_resnet18.onnx

mv dcn_resnet18.onnx ocdnet.onnx

# Download OCRnet onnx
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocrnet/deployable_v1.0/files?redirect=true&path=ocrnet_resnet50.onnx' -O ocrnet_resnet50.onnx

mv ocrnet_resnet50.onnx ocrnet.onnx

# Download OCRnet character_list
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocrnet/deployable_v1.0/files?redirect=true&path=character_list' -O character_list

mv character_list character_list.txt

# # Download command for ViT-based models:
# # Download OCDNet-ViT onnx
# wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocdnet/deployable_v2.0/files?redirect=true&path=ocdnet_fan_tiny_2x_icdar.onnx' -O ocdnet_fan_tiny_2x_icdar.onnx

# # Download OCRNet-ViT onnx
# wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocrnet/deployable_v2.0/files?redirect=true&path=ocrnet-vit.onnx' -O ocrnet-vit.onnx

# # Download OCRnet character_list
# wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocrnet/deployable_v2.0/files?redirect=true&path=character_list' -O character_list

```

#### **Compile the TensorRT OSS plugin libray (Optional)**:

**Notes**: If you're using TensorRT 8.6 and above, you can skip this step.

The OCDNet requires [`modulatedDeformConvPlugin`](https://github.com/NVIDIA/TensorRT/tree/release/8.6/plugin/modulatedDeformConvPlugin) for running with TensorRT

- Get TensorRT OSS repository
```shell
git clone -b release/8.6 https://github.com/NVIDIA/TensorRT.git
cd TensorRT
git submodule update --init --recursive
```

- Compile TensorRT `libnvinfer_plugin.so`:
```shell
mkdir build && cd build
# On X86 platform
cmake .. 
# On Jetson platform
# cmake .. -DTRT_LIB_DIR=/usr/lib/aarch64-linux-gnu/
make nvinfer_plugin -j4
```
**Notes**: You can use the [helper script](./tools/compile_trt_oss_jetson.sh) to compile TensorRT OSS.

- Copy the `libnvinfer_plugin.so` to the system library path
```shell
cp libnvinfer_plugin.so.8.6.0 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.5.1
# On Jetson platform:
# cp libnvinfer_plugin.so.8.6.0 /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8.5.2
```

#### **Generate TensorRT engine**:
Finally generate the TensorRT engine from trained OCDNet and OCRNet:
```shell
#Generate OCDNet engine with dynmaic batch size and max batch size is 4:
/usr/src/tensorrt/bin/trtexec --onnx=./ocdnet.onnx --minShapes=input:1x3x736x1280 --optShapes=input:1x3x736x1280 --maxShapes=input:4x3x736x1280 --fp16 --saveEngine=./ocdnet.fp16.engine

#Generate OCRNet engine with dynamic batch size and max batch size is 32:
/usr/src/tensorrt/bin/trtexec --onnx=./ocrnet.onnx --minShapes=input:1x1x32x100 --optShapes=input:32x1x32x100 --maxShapes=input:32x1x32x100 --fp16 --saveEngine=./ocrnet.fp16.engine

# #Generate engines for ViT-based models
# /usr/src/tensorrt/bin/trtexec --onnx=./ocdnet_fan_tiny_2x_icdar.onnx --minShapes=input:1x3x736x1280 --optShapes=input:1x3x736x1280 --maxShapes=input:1x3x736x1280 --fp16 --saveEngine=./ocdnet.fp16.engine

# /usr/src/tensorrt/bin/trtexec --onnx=./ocrnet-vit.onnx --minShapes=input:1x1x64x200 --optShapes=input:32x1x64x200 --maxShapes=input:32x1x64x200 --fp16 --saveEngine=./ocrnet.fp16.engine

```

### Building

- Clone the repository:
    ```shell
    git clone https://github.com/NVIDIA-AI-IOT/NVIDIA-Optical-Character-Detection-and-Recognition-Solution.git
    ```

- Compile the `libnvocdr.so`:
    ```shell
    cd NVIDIA-Optical-Character-Detection-and-Recognition-Solution
    make
    export LD_LIBRARY_PATH=$(pwd)
    ```
## Usage

### Use it in your application

To use nvOCDR in your C++ project, include the `nvOCRD.h` header file and link against the `nvOCDR` library. Here's an example code:

```c++
//test.cpp
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvocdr.h"

int main()
{

    // Init the nvOCDR lib
    // Please pay attention to the following parameters. You may need to change them according to different models.
    nvOCDRParam param;
    param.input_data_format = NHWC;
    param.ocdnet_trt_engine_path = (char *)"./ocdnet.fp16.engine";
    param.ocdnet_infer_input_shape[0] = 3;
    param.ocdnet_infer_input_shape[1] = 736;
    param.ocdnet_infer_input_shape[2] = 1280;
    param.ocdnet_binarize_threshold = 0.1;
    param.ocdnet_polygon_threshold = 0.3;
    param.ocdnet_max_candidate = 200;
    param.ocdnet_unclip_ratio = 1.5;
    param.ocrnet_trt_engine_path = (char *)"./ocrnet.fp16.engine";
    param.ocrnet_dict_file = (char *)"./character_list.txt";
    param.ocrnet_infer_input_shape[0] = 1;
    param.ocrnet_infer_input_shape[1] = 32;
    param.ocrnet_infer_input_shape[2] = 100;
    // uncomment if you're using attention-based models:
    // param.ocrnet_decode = Attention;
    nvOCDRp nvocdr_ptr = nvOCDR_init(param);

    // Load the input
    const char* img_path = "./test.jpg";
    cv::Mat img = cv::imread(img_path);
    nvOCDRInput input;
    input.device_type = GPU;
    input.shape[0] = 1;
    input.shape[1] = img.size().height;
    input.shape[2] = img.size().width;
    input.shape[3] = 3;
    size_t item_size = input.shape[1] * input.shape[2] * input.shape[3] * sizeof(uchar);
    cudaMalloc(&input.mem_ptr, item_size);
    cudaMemcpy(input.mem_ptr, reinterpret_cast<void*>(img.data), item_size, cudaMemcpyHostToDevice);

    // Do inference
    nvOCDROutputMeta output;
    nvOCDR_inference(input, &output, nvocdr_ptr);

    // Print the output
    int offset = 0;
    for(int i = 0; i < output.batch_size; i++)
    {
        for(int j = 0; j < output.text_cnt[i]; j++)
        {
            printf("%d : %s, %ld\n", i, output.text_ptr[offset].ch, strlen(output.text_ptr[offset].ch));
            offset += 1;
        }
    }

    // Destroy the resoures
    free(output.text_ptr);
    cudaFree(input.mem_ptr);
    nvOCDR_deinit(nvocdr_ptr);

    return 0;
}
```

You can compile the code with the command:
```shell
g++ ./test.cpp -I./include -L./ -I/usr/include/opencv4/ -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lopencv_core -lopencv_imgcodecs -lnvocdr -o test
```

### Use nvOCDR in DeepStream SDK

For more information on how to use nvOCDR in DeepStream, see the [documentation](https://docs.nvidia.com/tao/tao-toolkit/text/ds_tao/nvocdr_ds.html).

### Use nvOCDR in Triton

For more information on how to use nvOCDR in Triton, see the [documentation](./triton/README.md).

### Use OCRNet with attention module

The ViT-based OCRNet models released on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/ocrnet) (**deployable 2.0** and **deployable 2.1**) come with attention module which require attention decoding method. One can enable attention decoding by the following steps:

- In C++ application:

    ```c++
    nvOCDRParam param;
    param.ocrnet_decode = Attention;
    ```

- In DeepStream:

    ```shell
    customlib-props="ocrnet-decode:Attention"
    ```

- In Triton (in `models/nvOCDR/spec.json`):

    ```json
    "ocrnet_decode": "Attention"
    ```

### API Reference

For more information about nvOCDR API, see the [API reference](doc/nvOCDR.md)


## License

By cloning or downloading nvOCDR, you agree to terms of the [nvOCDR EULA](./LICENSE).
