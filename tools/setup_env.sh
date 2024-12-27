#!/bin/bash

apt install -y lsb-release software-properties-common libopencv-dev libboost-program-options-dev libgtest-dev libgoogle-glog-dev
wget https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh; chmod +x /tmp/llvm.sh; /tmp/llvm.sh 18

apt install -y clang-format-18