#!/usr/bin/env bash
export LD_LIBRARY_PATH=/home/ant/miniconda3/envs/curbnet/lib/python3.8/site-packages/spconv:${LD_LIBRARY_PATH}

/home/ant/miniconda3/envs/curbnet/bin/python -u sweep_z_shift_nrs.py "$@"
