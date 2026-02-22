#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CV_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# Allocator choice:
# Option A (default): cudaMallocAsync allocator (usually reduces fragmentation in long multi-GPU runs)
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
# If you want to revert to native caching allocator:
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024,garbage_collection_threshold:0.8

python -m torch.distributed.run --nproc_per_node=4 Geo_train.py \
    --config_path ./config/config_Geo_localGT.yml \
    --dist \
    --GPU_ids 0,1,2,3