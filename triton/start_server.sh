# docker run --rm --net=host --gpus=all -p 8000:8000 -v $PWD:/nvocdr -v $PWD/build/install/backends:/models nvocdr_triton_server \
#           tritonserver --model-repository /models --backend-config nvocdr,spec=/nvocdr/param.json --log-verbose 1 

docker run --rm --net=host --gpus=all -it -p 8000:8000 -v $PWD:/nvocdr -v $PWD/build/install/backends:/models nvocdr_triton_server bash

