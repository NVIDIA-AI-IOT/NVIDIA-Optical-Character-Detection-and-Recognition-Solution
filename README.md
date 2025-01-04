# nvOCDR

## build 

1. clone the repository 
2. `docker run --gpus=all -v {repo_dir}:/workdir --rm -it --privileged --net=host nvcr.io/nvidia/tensorrt:23.11-py3 bash`
3. run under repo root: `bash scripts/setup_build_env.sh`
4. run under repo root: `bash scripts/download_models.sh`
5. run under repo root: `cmake -B build -S .`
6. `cd build; make -j`
7. `samples/c++/sample_inference ...` to run,  `./build/samples/c++/sample_inference --help` to check the options
   > ./build/samples/c++/sample_inference --ocd_model ./onnx_models/dcn_resnet18.engine --ocr_model ./onnx_models/ocrnet_resnet50.engine --image ./samples/test_img/scene_text.jpg