#!/bin/bash

# This script builds a ready-to-use docker image of nvOCDR DeepStream sample
if [ $# -ne 7 ]; then
    echo -e "Usage: bash build_docker.sh [OCD_MODEL_PATH] [OCR_MODEL_PATH] [OCR_VOC_PATH] [OCD_HEIGHT] [OCD_WIDTH] [OCD_MAX_BS] [DEVICE]"
    exit 1
fi

cd ../../
mkdir -p models
cp $1 models/ocdnet.onnx
cp $2 models/ocrnet.onnx
cp $3 models/character_list

docker build . -t nvocdr_ds_sample:v1.0 -f NVIDIA-Optical-Character-Detection-and-Recognition-Solution/deepstream/Dockerfile \
--build-arg OCD_MODEL_PATH=models/ocdnet.onnx \
--build-arg OCR_MODEL_PATH=models/ocrnet.onnx \
--build-arg OCR_VOC_PATH=models/character_list \
--build-arg OCD_H=$4 \
--build-arg OCD_W=$5 \
--build-arg OCD_MAX_BS=$6 \
--build-arg DEVICE=$7
