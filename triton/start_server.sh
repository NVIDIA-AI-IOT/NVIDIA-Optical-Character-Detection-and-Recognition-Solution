docker run --rm --net=host -v $PWD/build/install/backends:/models nvcr.io/nvidia/tritonserver:24.12-py3 tritonserver \
         --model-repository /models \
         --backend-config nvocdr,spec=xxx.json
