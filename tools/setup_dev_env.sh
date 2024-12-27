#!/bin/bash

CLANG_VERSION=18
apt install -y lsb-release software-properties-common 
wget https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh; chmod +x /tmp/llvm.sh; /tmp/llvm.sh $CLANG_VERSION

apt install -y clang-format-$CLANG_VERSION clang-tidy-$CLANG_VERSION