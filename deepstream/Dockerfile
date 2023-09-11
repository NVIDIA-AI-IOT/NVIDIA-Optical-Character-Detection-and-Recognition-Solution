FROM nvcr.io/nvidia/deepstream:6.2-triton

# Copy the content of repository 
RUN mkdir -p /opt/nvidia/nvOCDR/ocdr
COPY NVIDIA-Optical-Character-Detection-and-Recognition-Solution /opt/nvidia/nvOCDR/ocdr/

# Install opencv
RUN apt update && apt install -y libopencv-dev

# Install TensorRT plugin with ModulatedDeformableConvolution
RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.0/local_repos/nv-tensorrt-local-repo-ubuntu2004-8.6.0-cuda-11.8_1.0-1_amd64.deb && \
    dpkg-deb -xv nv-tensorrt-local-repo-ubuntu2004-8.6.0-cuda-11.8_1.0-1_amd64.deb debs && \
    cd debs/var/nv-tensorrt-local-repo-ubuntu2004-8.6.0-cuda-11.8 && \
    dpkg-deb  -xv libnvinfer-plugin8_8.6.0.12-1+cuda11.8_amd64.deb deb_file && \
    cp deb_file/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.6.0  /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.5.2

# Compile nvOCDR core lib
RUN cd /opt/nvidia/nvOCDR/ocdr && make -j8
ENV LD_LIBRARY_PATH=/opt/nvidia/nvOCDR/ocdr:$LD_LIBRARY_PATH

# Compile DeepStream intermedia library
ENV CUDA_VER=11.8
RUN cd /opt/nvidia/nvOCDR/ocdr/deepstream && make

# Generate TensorRT engine
RUN mkdir -p /opt/nvidia/nvOCDR/models
ARG OCD_MODEL_PATH=models/ocdnet.onnx
COPY ${OCD_MODEL_PATH} /opt/nvidia/nvOCDR/models/ocdnet.onnx
ARG OCR_MODEL_PATH=models/ocrnet.onnx
COPY ${OCR_MODEL_PATH} /opt/nvidia/nvOCDR/models/ocrnet.onnx
ARG OCR_VOC_PATH=models/character_list
COPY ${OCR_VOC_PATH} /opt/nvidia/nvOCDR/models/character_list
ARG OCD_H=736
ARG OCD_W=1280
ARG OCD_MAX_BS=4
ARG DEVICE=0

RUN mkdir -p /opt/nvidia/nvOCDR/engines 
RUN /usr/src/tensorrt/bin/trtexec --device=${DEVICE} --onnx=/opt/nvidia/nvOCDR/models/ocdnet.onnx \
--minShapes=input:1x3x${OCD_H}x${OCD_W} --optShapes=input:${OCD_MAX_BS}x3x${OCD_H}x${OCD_W} \
--maxShapes=input:${OCD_MAX_BS}x3x${OCD_H}x${OCD_W} \
--fp16 --saveEngine=/opt/nvidia/nvOCDR/engines/ocdnet.fp16.engine
RUN /usr/src/tensorrt/bin/trtexec --device=${DEVICE} --onnx=/opt/nvidia/nvOCDR/models/ocrnet.onnx \
--minShapes=input:1x1x32x100 --optShapes=input:1x1x32x100 --maxShapes=input:32x1x32x100 --fp16 \
--saveEngine=/opt/nvidia/nvOCDR/engines/ocrnet.fp16.engine

# Generate gst-launch-1.0 command
WORKDIR /opt/nvidia/nvOCDR
# use new nvstreammux to avoid processing images in this plugin
ENV USE_NEW_NVSTREAMMUX=yes
RUN echo "CUDA_VISIBLE_DEVICES=${DEVICE} gst-launch-1.0 filesrc location=\$1 ! qtdemux ! h264parse ! nvv4l2decoder ! \
        m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 ! \
        nvdsvideotemplate customlib-name=/opt/nvidia/nvOCDR/ocdr/deepstream/libnvocdr_impl.so \
        customlib-props="ocdnet-engine-path:/opt/nvidia/nvOCDR/engines/ocdnet.fp16.engine" \
        customlib-props="ocdnet-input-shape:3,736,1280" \
        customlib-props="ocdnet-binarize-threshold:0.1" \
        customlib-props="ocdnet-polygon-threshold:0.3" \
        customlib-props="ocdnet-max-candidate:200" \
        customlib-props="ocrnet-engine-path:/opt/nvidia/nvOCDR/engines/ocrnet.fp16.engine" \
        customlib-props="ocrnet-dict-path:/opt/nvidia/nvOCDR/models/character_list" \
        customlib-props="ocrnet-input-shape:1,32,100" ! \
        nvmultistreamtiler rows=1 columns=1 width=1280 height=720 ! nvvideoconvert ! nvdsosd ! \
        nvvideoconvert ! 'video/x-raw(memory:NVMM),format=I420' ! nvv4l2h264enc ! h264parse ! \
        mux.video_0 qtmux name=mux ! filesink location=output.mp4" > run_demo.sh