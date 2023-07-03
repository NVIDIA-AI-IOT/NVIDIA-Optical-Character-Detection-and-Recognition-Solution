#!/bin/bash

set -Ee

if [ $# -ne 7 ]; then
    echo -e "Usage: \n bash setup.sh [OCD input height] [OCD input width] [OCD input max batchsize] [DEVICE] [ocd onnx path> [ocr onnx path] [ocr character list path]\n \n For example: \n      
    bash setup_triton_server.sh 736 1280 4 0 model/ocd.onnx model/ocr.onnx model/ocr_character_list\n\n 
    Available Options:    
    OCD input Height:           the input height of ocd tensorRT engine   
    OCD input width:            the input width of ocd tensorRT engine   
    OCD input max batchsize:    the max batch size of ocd tensorRT engine  
    DEVICE:                     gpu idx on which to build trt engine
    OCD_MODEL_PATH:             path to the ocdnet onnx model
    OCD_MODEL_PATH:             path to the ocrnet onnx model
    OCR_VOC_PATH:               path to the ocrnet character list"
    exit 1
fi

echo "Set input size to HxW: ${1}x${2} and max batch size: ${3} for ocd tensorRT engine"

cd ../../
mkdir -p models
cp ${5} models/ocdnet.onnx
cp ${6} models/ocrnet.onnx
cp ${7} models/character_list


# build triton server docker image
docker build . -f NVIDIA-Optical-Character-Detection-and-Recognition-Solution/triton/Triton_Server.Dockerfile -t nvcr.io/nvidian/tao/nvocdr_triton_server:v1.0 \
--build-arg OCD_H=${1} \
--build-arg OCD_W=${2} \
--build-arg OCD_MAX_BS=${3} \
--build-arg DEVICE=${4} \
--build-arg OCD_MODEL_PATH=models/ocdnet.onnx \
--build-arg OCR_MODEL_PATH=models/ocrnet.onnx \
--build-arg OCR_VOC_PATH=models/character_list
