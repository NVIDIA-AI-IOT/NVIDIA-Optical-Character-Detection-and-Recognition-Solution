#!/bin/bash

apt update
apt install -y libopencv-dev libboost-program-options-dev libgtest-dev libgoogle-glog-dev rapidjson-dev

# for python bind
pip install opencv-python