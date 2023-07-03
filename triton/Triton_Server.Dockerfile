ARG BASE_TRITON_IMAGE=nvcr.io/nvidia/tritonserver:23.02-py3
FROM ${BASE_TRITON_IMAGE} AS build

# step1: nvocdr triton sample workdir 
RUN mkdir -p /opt/nvocdr && cd /opt/nvocdr

# step2: TODO get onnx from ngc
RUN mkdir -p /opt/nvocdr/onnx_model 

ARG OCD_MODEL_PATH=models/ocdnet.onnx
ARG OCR_MODEL_PATH=models/ocrnet.onnx
ARG OCR_VOC_PATH=models/character_list

COPY ${OCD_MODEL_PATH} /opt/nvocdr/onnx_model/ocdnet.onnx
COPY ${OCR_MODEL_PATH} /opt/nvocdr/onnx_model/ocrnet.onnx
COPY ${OCR_VOC_PATH} /opt/nvocdr/onnx_model/character_list

RUN ls -l /opt/nvocdr/onnx_model/
# step3: install deformable-conv trt plugin
RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.0/local_repos/nv-tensorrt-local-repo-ubuntu2004-8.6.0-cuda-11.8_1.0-1_amd64.deb && \
    dpkg-deb -xv nv-tensorrt-local-repo-ubuntu2004-8.6.0-cuda-11.8_1.0-1_amd64.deb debs && \
    cd debs/var/nv-tensorrt-local-repo-ubuntu2004-8.6.0-cuda-11.8 && \
    dpkg-deb  -xv libnvinfer-plugin8_8.6.0.12-1+cuda11.8_amd64.deb deb_file && \
    cp deb_file/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.6.0  /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.5.3
    
# step4: build trt engines
ARG OCD_H=736
ARG OCD_W=1280
ARG OCD_MAX_BS=4
ARG DEVICE=0

RUN mkdir -p /opt/nvocdr/engines 
RUN /usr/src/tensorrt/bin/trtexec --device=${DEVICE} --onnx=/opt/nvocdr/onnx_model/ocdnet.onnx --minShapes=input:1x3x${OCD_H}x${OCD_W} --optShapes=input:${OCD_MAX_BS}x3x${OCD_H}x${OCD_W} --maxShapes=input:${OCD_MAX_BS}x3x${OCD_H}x${OCD_W} --fp16 --saveEngine=/opt/nvocdr/engines/ocdnet.fp16.engine
RUN /usr/src/tensorrt/bin/trtexec --device=${DEVICE} --onnx=/opt/nvocdr/onnx_model/ocrnet.onnx --minShapes=input:1x1x32x100 --optShapes=input:1x1x32x100 --maxShapes=input:32x1x32x100 --fp16 --saveEngine=/opt/nvocdr/engines/ocrnet.fp16.engine

# step5: install opencv
RUN apt-get update && apt-get install libgl1-mesa-glx --yes && apt-get install libopencv-dev --yes


# step6: COPY ocdr repo and install python packages
COPY NVIDIA-Optical-Character-Detection-and-Recognition-Solution /opt/nvocdr/ocdr
RUN cd /opt/nvocdr/ocdr/triton && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install nvidia-pyindex && \
    python3 -m pip install -r requirements-pip.txt


# change the ocd input size in spec.json
ENV OCD_INPUT_H=${OCD_H}
ENV OCD_INPUT_W=${OCD_W}
RUN sed -i "s|OCD_INPUT_H|${OCD_INPUT_H}|g" /opt/nvocdr/ocdr/triton/models/nvOCDR/spec.json && \
    sed -i "s|OCD_INPUT_W|${OCD_INPUT_W}|g" /opt/nvocdr/ocdr/triton/models/nvOCDR/spec.json

# step7: build nvocdr lib    
RUN cd /opt/nvocdr/ocdr/triton && make -j8

ENV PYTHONPATH="/opt/nvocdr/ocdr/triton:${PYTHONPATH}"
WORKDIR /opt/nvocdr



