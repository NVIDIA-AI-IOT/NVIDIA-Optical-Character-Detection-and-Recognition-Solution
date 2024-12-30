# nvOCDR

## build 

1. clone the repository 
2. `docker run --gpus=all -v {repo_dir}:/workdir --rm -it --privileged --net=host nvcr.io/nvidia/tensorrt:23.11-py3 bash`
3. run under repo root: `bash scripts/setup_build_env.sh`
4. run under repo root: `bash scripts/download_models.sh`
5. run under repo root: `cmake -B build -S .`
6. `cd build; make -j`
7. `samples/c++/sample_inference ...` to run, example as below
   > build/samples/c++/sample_inference --ocd_model ./onnx_models/dcn_resnet18.engine  --ocr_model ./onnx_models/ocrnet_resnet50.engine --max_candidates 100 --image ./samples/test_img/scene_text.jpg --text_polygon_thresh 0.2 --show_score 1 --debug_log 0 --strategy smart --rec_all_direction 0 --ocd_type normal --ocr_type CTC --debug_image 0 --binary_lower_bound 0.0 --binary_upper_bound 0.1