name=cylinder_asym_networks_12gb
gpuid=0

export LD_LIBRARY_PATH=/home/ant/miniconda3/envs/curbnet/lib/python3.8/site-packages/spconv:${LD_LIBRARY_PATH}

CUDA_VISIBLE_DEVICES=${gpuid} python -u train_cylinder_asym_0.2.py \
  -y config/semantickitti-curb_0.2_12gb.yaml \
2>&1 | tee logs_dir/${name}_curb_seq0.2.txt
