# nvOCDR

## build 

1. clone the repository 
2. `docker run --gpus=all -v {repo_dir}:/workdir --rm -it --privileged --net=host nvcr.io/nvidia/tensorrt:24.12-py3 bash`
3. run under repo root: `bash scripts/setup_build_env.sh`
4. run under repo root: `bash scripts/download_models.sh`
5. run under repo root: `cmake -B build -S .`
6. `cd build; make -j`
7. `samples/c++/sample_inference ...` to run,  `./build/samples/c++/sample_inference --help` to check the options
   * simple run dcn_resnet18 + ocrnet_resnet50: `./build/samples/c++/sample_inference --ocd_model ./onnx_models/dcn_resnet18.engine --ocr_model ./onnx_models/ocrnet_resnet50.engine --image ./samples/test_img/scene_text.jpg`
   
   * super_resolution pcb + mixnet + CLIP: `./build/samples/c++/sample_inference --image ./samples/test_img/super_resolution.jpg --ocr_model onnx_models/clip_visual.engine,onnx_models/clip_text.engine --ocd_model onnx_models/mixnet.engine --ocd_type mixnet  --ocr_type CLIP --clip_vocab_file onnx_models/bpe_simple_vocab_16e6.txt --binary_lower_bound 0.058 --binary_upper_bound 0.54 --debug_image 1 --strategy resize_full --max_candidates 1000 --rec_all_direction 1`


# To be continued...