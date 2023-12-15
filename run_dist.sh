#!/usr/bin/env bash

# 设置 Anaconda 环境
. "/home/ps/anaconda3/etc/profile.d/conda.sh"
conda activate time_series

# 设置 DeepSpeed 环境变量
export PDSH_RCMD_TYPE=ssh

# 启动 DeepSpeed 训练
deepspeed --num_gpus=4 main.py 