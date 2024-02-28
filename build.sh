#!/bin/bash

mkdir build
cd build
ROCM_PATH=/opt/rocm CXX=hipcc cmake -DBUILD_BENCHMARK=OFF -DBUILD_TEST=ON -DAMDGPU_TARGETS=gfx1101 ../.
make -j test_device_adjacent_difference