#!/bin/bash

# Cài đặt jetson-inference
git clone --recursive https://github.com/dusty-nv/jetson-inference.git
cd jetson-inference
mkdir build
cd build
cmake ..
make -j$(nproc)
yes | sudo make install  # Tự động trả lời OK cho quá trình cài đặt Pytorch

echo "Cài đặt jetson-inference hoàn tất!"
