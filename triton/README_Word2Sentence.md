# nvOCDR word2sentence

- clone the repository
  ```
  git clone https://github.com/NVIDIA-AI-IOT/NVIDIA-Optical-Character-Detection-and-Recognition-Solution.git 
  ```

- download the onnx models of OCDnet and OCRnet
  ```
  mkdir NVIDIA-Optical-Character-Detection-and-Recognition-Solution/onnx_models
  cd NVIDIA-Optical-Character-Detection-and-Recognition-Solution/onnx_models

  # Download OCDNet-ViT onnx
  wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocdnet/deployable_v2.0/files?redirect=true&path=ocdnet_fan_tiny_2x_icdar.onnx' -O ocdnet_fan_tiny_2x_icdar.onnx

  # Download OCRNet-ViT onnx
  wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocrnet/deployable_v2.0/files?redirect=true&path=ocrnet-vit.onnx' -O ocrnet-vit.onnx

  # Download OCRnet character_list
  wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/ocrnet/deployable_v2.0/files?redirect=true&path=character_list' -O character_list
  ```

- Build nvocdr docker images

 
    ```
    cd NVIDIA-Optical-Character-Detection-and-Recognition-Solution/triton
    
    bash setup_triton_server.sh [ocd onnx path] [ocr onnx path] [ocr character list path]

    # For example
    bash setup_triton_server.sh ../onnx_models/ocdnet_fan_tiny_2x_icdar.onnx ../onnx_models/ocrnet-vit.onnx ../onnx_models/character_list
    ```

- Start a nvocdr container  
    __Note:  ocdnet_fan_tiny_2x_icdar.onnx only support batch size 1__
    ```
    docker run -it --rm --net=host --gpus all --shm-size 8g nvcr.io/nvidian/tao/nvocdr_triton_server:v2.0 bash

    # build the tensorRT engine for OCDNet and OCRnet
    cd /opt/nvocdr/ocdr/triton

    bash build_engine.sh [OCD input height] [OCD input width] [OCD input max batchsize] [GPU idx]

    # OCDNet-ViT onnx model only support batch_size=1, For example
    bash build_engine.sh 736 1280 1 0


 - Inference for high resolution images  
      nvocdr triton can support hight resolution images as input larger than 736x1280x3 (HWC). you can change the spec file in `models/nvOCDR/spec.json` to support the high resolution images inference.
      ```
      # to support high resolution images
      is_high_resolution_input: true
      ```
      __Note: high resolution image inference only support batch size 1__


      
- Run word2sentence sample 
  ```
  cd NVIDIA-Optical-Character-Detection-and-Recognition-Solution/triton 
  python3 sample_word2sentence.py -d ../c++_samples/paper_img -bs 1 -c ./models/nvOCDR/spec.json
  ```