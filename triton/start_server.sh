docker run --rm --net=host -v $PWD:/nvocdr -v $PWD/build/install/backends:/models nvocdr_triton_server tritonserver --model-repository /models --backend-config nvocdr,spec=/nvocdr/param.json 

# docker run --rm --net=host -it -v $PWD:/nvocdr -v $PWD/build/install/backends:/models nvocdr_triton_server bash

