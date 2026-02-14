#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CV_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python -m torch.distributed.run --nproc_per_node=2 Geo_train.py \
    --config_path ./config/config_GeoRefKV_2GPU.yml \
    --dist \
    --GPU_ids 0,1