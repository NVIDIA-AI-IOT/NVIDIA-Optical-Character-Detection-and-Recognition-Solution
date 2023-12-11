#!/bin/bash

set -Ee

WORKDIR=`pwd`
echo ${WORKDIR}

# build triton client docker image
docker build . -f ./Triton_Client.Dockerfile -t nvcr.io/nvidian/tao/nvocdr_triton_client:v2.0
