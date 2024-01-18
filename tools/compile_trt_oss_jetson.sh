# Get the GPU_ARCH:
GPU_ARCHS=$(python get_archs.py)

# Clone the TRT OSS repo:
git clone -b release/8.6 https://github.com/NVIDIA/TensorRT.git
cd TensorRT
git submodule update --init --recursive

# Compile the library
mkdir build && cd build
cmake .. -DGPU_ARCHS=$GPU_ARCHS -DTRT_LIB_DIR=/usr/lib/aarch64-linux-gnu/
make nvinfer_plugin -j4