ARG BASE_TRITON_IMAGE=nvcr.io/nvidia/tritonserver:24.09-py3 
FROM ${BASE_TRITON_IMAGE} AS build

# step1: nvocdr triton sample workdir 
RUN mkdir -p /opt/nvocdr && cd /opt/nvocdr

# step2: TODO get onnx from ngc
RUN mkdir -p /opt/nvocdr/onnx_model 

ARG OCD_MODEL_PATH=models/ocdnet_vit.onnx
ARG OCR_MODEL_PATH=models/ocrnet_vit.onnx
ARG OCR_VOC_PATH=models/character_list

COPY ${OCD_MODEL_PATH} /opt/nvocdr/onnx_model/ocdnet_vit.onnx
COPY ${OCR_MODEL_PATH} /opt/nvocdr/onnx_model/ocrnet_vit.onnx
COPY ${OCR_VOC_PATH} /opt/nvocdr/onnx_model/character_list

RUN ls -l /opt/nvocdr/onnx_model/

# step3: install opencv
RUN apt-get update && apt-get install libgl1-mesa-glx --yes && apt-get install libopencv-dev --yes


# step4: COPY ocdr repo and install python packages
COPY . /opt/nvocdr/ocdr
RUN cd /opt/nvocdr/ocdr/triton && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install nvidia-pyindex && \
    python3 -m pip install -r requirements-pip.txt



# step5: build nvocdr lib    
RUN cd /opt/nvocdr/ocdr/triton && make -j8

ENV PYTHONPATH="/opt/nvocdr/ocdr/triton:${PYTHONPATH}"
WORKDIR /opt/nvocdr
