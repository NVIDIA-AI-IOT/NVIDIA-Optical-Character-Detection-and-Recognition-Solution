
#!/bin/bash

set -Ee

if [ $# -ne 4 ]; then
    echo -e "Usage: \n bash build_engine.sh [OCD input height] [OCD input width] [OCD input max batchsize] [DEVICE] \n For example: \n
        bash build_engine.sh 736 1280 4 0"
    exit 1      
fi
# build trt engines
OCD_H=${1}
OCD_W=${2}
OCD_MAX_BS=${3}
DEVICE=${4}
mkdir -p /opt/nvocdr/engines
/usr/src/tensorrt/bin/trtexec --device=${DEVICE} --onnx=/opt/nvocdr/onnx_model/ocdnet.onnx --minShapes=input:1x3x${OCD_H}x${OCD_W} --optShapes=input:${OCD_MAX_BS}x3x${OCD_H}x${OCD_W} --maxShapes=input:${OCD_MAX_BS}x3x${OCD_H}x${OCD_W} --fp16 --saveEngine=/opt/nvocdr/engines/ocdnet.fp16.engine
/usr/src/tensorrt/bin/trtexec --device=${DEVICE} --onnx=/opt/nvocdr/onnx_model/ocrnet.onnx --minShapes=input:1x1x32x100 --optShapes=input:1x1x32x100 --maxShapes=input:32x1x32x100 --fp16 --saveEngine=/opt/nvocdr/engines/ocrnet.fp16.engine

# change the ocd input size in spec.json
OCD_INPUT_H=${OCD_H}
OCD_INPUT_W=${OCD_W}
sed -i "s|OCD_INPUT_H|${OCD_H}|g" /opt/nvocdr/ocdr/triton/models/nvOCDR/spec.json
sed -i "s|OCD_INPUT_W|${OCD_W}|g" /opt/nvocdr/ocdr/triton/models/nvOCDR/spec.json
