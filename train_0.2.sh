name=cylinder_asym_networks
gpuid=0

CUDA_VISIBLE_DEVICES=${gpuid}  python -u train_cylinder_asym_0.2.py \
2>&1 | tee logs_dir/${name}_curb_seq0.2.txt



#### Evaluation
# python demo_folder_focal.py 
# --demo-folder /home/XXX 
# --save-folder /home/XXX 
# --demo-label-folder /home/XXX