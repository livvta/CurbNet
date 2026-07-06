#!/bin/bash
# CurbNet 交互式推理可视化 - 启动脚本
# 自动设置 spconv 库路径

export LD_LIBRARY_PATH=/home/ant/miniconda3/envs/curbnet/lib/python3.8/site-packages/spconv:${LD_LIBRARY_PATH}

python -u visualize_demo_interactive.py "$@"
