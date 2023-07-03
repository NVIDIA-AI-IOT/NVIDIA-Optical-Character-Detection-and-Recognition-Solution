# How to integrate nvOCDR into Triton

- clone the repository
  ```
  git clone https://github.com/NVIDIA-AI-IOT/NVIDIA-Optical-Character-Detection-and-Recognition-Solution.git && \
  cd NVIDIA-Optical-Character-Detection-and-Recognition-Solution/triton
  ```
- Build Triton Server docker images

 
    ```
    bash setup_triton_server.sh [OCD input height] [OCD input width] [OCD input max batchsize] [DEVICE] [ocd onnx path> [ocr onnx path] [ocr character list path]

    # For example
    bash setup_triton_server.sh 736 1280 4 0 model/ocd.onnx model/ocr.onnx model/ocr_character_list
    ```

- Build Triton Client docker images
  ```
  bash setup_triton_client.sh
  ```
- Start a triton server container
    ```
    docker run -it --rm --net=host --gpus all --shm-size 8g nvcr.io/nvidian/tao/nvocdr_triton_server:v1.0 bash

    CUDA_VISIBLE_DEVICES=<gpu idx> tritonserver --model-repository /opt/nvocdr/ocdr/triton/models/
    ```
    - Inference for high resolution images  
      nvocdr triton can support hight resolution images as input such as 4000x4000. you can change the spec file in `models/nvOCDR/spec.json` to support the high resolution images inference.
      ```
      # to support high resolution images
      is_high_resolution_input: true
      ```
      __Note: high resolution image inference only support batch size 1__


      
- Start a triton client container
  ```
  docker run -it --rm -v <path to images dir>:<path to images dir>  --net=host nvcr.io/nvidian/tao/nvocdr_triton_client:v1.0 bash

  python3 client.py -d <path to images dir> -bs 1
  ```

  args of `client.py`:

  `-d`: path to image directory  
  `-bs`: infer batch size