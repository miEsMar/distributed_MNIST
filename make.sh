#!/bin/sh


module purge
ml torchcpu

mkdir -p ./build

cmake \
    -DCMAKE_PREFIX_PATH=/gpfs/projects/bsc85/bsc488161/pytorch/torch/ \
    -S . -B build
cmake --build build


