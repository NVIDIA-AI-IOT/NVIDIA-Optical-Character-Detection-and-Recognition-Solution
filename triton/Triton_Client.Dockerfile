ARG BASE_TRITON_IMAGE=nvcr.io/nvidia/tritonserver:23.11-py3-sdk 
FROM ${BASE_TRITON_IMAGE} AS build



# pip install
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install opencv-python && \
    python3 -m pip install pynvjpeg && \
    python3 -m pip install pycuda==2022.2.2 

# nvocdr triton sample workdir 
RUN mkdir -p /opt/nvocdr && cd /opt/nvocdr

COPY . /opt/nvocdr/triton

WORKDIR /opt/nvocdr/triton

