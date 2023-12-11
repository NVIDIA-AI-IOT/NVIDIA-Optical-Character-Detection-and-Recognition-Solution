#!/bin/bash

set -Ee

if [ $# -ne 3 ]; then
    echo -e "Usage: \n bash setup.sh [ocd onnx path] [ocr onnx path] [ocr character list path]\n \n For example: \n      
    bash setup_triton_server.sh model/ocd.onnx model/ocr.onnx model/ocr_character_list\n 
    Available Options:    
    OCD_MODEL_PATH:             path to the ocdnet onnx model
    OCR_MODEL_PATH:             path to the ocrnet onnx model
    OCR_VOC_PATH:               path to the ocrnet character list"
    exit 1
fi


if [[ "$PWD" =~ ".*NVIDIA-Optical-Character-Detection-and-Recognition-Solution/triton$" ]]; then
  echo "This script must be run from the /<your-path>/NVIDIA-Optical-Character-Detection-and-Recognition-Solution/triton directory!"
  exit 1
fi

mkdir -p models

if [ ! -f $PWD/${1} ]; then
    echo "can not find  $PWD/${1}"
    exit 1
fi

if [ ! -f $PWD/${2} ]; then
    echo "can not find  $PWD/${2}"
    exit 1
fi

if [ ! -f $PWD/${3} ]; then
    echo "can not find  $PWD/${3}"
    exit 1
fi

cp $PWD/${1} models/ocdnet_vit.onnx
cp $PWD/${2} models/ocrnet_vit.onnx
cp $PWD/${3} models/character_list


# build triton server docker image
echo "Start to build triton server docker image..."
cd ../
docker build . -f ./triton/Triton_Server.Dockerfile -t nvcr.io/nvidian/tao/nvocdr_triton_server:v2.0 \
--build-arg OCD_MODEL_PATH=./triton/models/ocdnet_vit.onnx \
--build-arg OCR_MODEL_PATH=./triton/models/ocrnet_vit.onnx \
--build-arg OCR_VOC_PATH=./triton/models/character_list
