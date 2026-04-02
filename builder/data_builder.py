# -*- coding:utf-8 -*-
# author: CurbNet
# @file: data_builder.py 

import torch
from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV
from dataloader.pc_dataset import get_pc_model_class


def build(dataset_config,
          train_dataloader_config,
          val_dataloader_config,
          grid_size=[480, 360, 32]):
    data_path = train_dataloader_config["data_path"]
    train_imageset = train_dataloader_config["imageset"]
    val_imageset = val_dataloader_config["imageset"]
    train_ref = train_dataloader_config["return_ref"]
    val_ref = val_dataloader_config["return_ref"]

    label_mapping = dataset_config["label_mapping"]

    SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])

    nusc=None
    if "nusc" in dataset_config['pc_dataset_type']:
        from nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)

    train_pt_dataset = SemKITTI(data_path, imageset=train_imageset,
                                return_ref=train_ref, label_mapping=label_mapping, nusc=nusc)
    val_pt_dataset = SemKITTI(data_path, imageset=val_imageset,
                              return_ref=val_ref, label_mapping=label_mapping, nusc=nusc)
    
    print("++",train_pt_dataset)

    train_dataset = get_model_class(dataset_config['dataset_type'])(
        train_pt_dataset,
        grid_size=grid_size,
        flip_aug=True,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
        rotate_aug=True,
        scale_aug=True,
        transform_aug=True
    )

    val_dataset = get_model_class(dataset_config['dataset_type'])(
        val_pt_dataset,
        grid_size=grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
    )
    # print('11111111111111111111111111111111111')
    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_dataloader_config["batch_size"],
                                                       collate_fn=collate_fn_BEV,
                                                       shuffle=train_dataloader_config["shuffle"],
                                                       num_workers=train_dataloader_config["num_workers"])
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=collate_fn_BEV,
                                                     shuffle=val_dataloader_config["shuffle"],
                                                     num_workers=val_dataloader_config["num_workers"])
    # print('222222222222222222222222222222222')

    return train_dataset_loader, val_dataset_loader



import yaml
import numpy as np
from pathlib import Path

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

# def calculate_class_counts(yaml_path, dataset_path):
#     yaml_data = load_yaml(yaml_path)
#     learning_map = yaml_data['learning_map']
#     train_sequences = yaml_data['split']['train']

#     # 鍒濆鍖栫被鍒鏁颁负0
#     max_label = max(learning_map.values())
#     class_counts = np.zeros(max_label + 1, dtype=np.int64)

#     for seq in train_sequences:
#         seq_path = Path(dataset_path) / f"{seq:02d}" / "labels"
#         for label_file in seq_path.rglob('*.label'):
#             # 璇诲彇姣忎釜鐐圭殑鏍囩
#             labels = np.fromfile(str(label_file), dtype=np.uint32)
#             # 鑾峰彇浣?16 浣嶇殑璇箟鏍囩
#             semantic_labels = labels & 0xFFFF
#             # 瀵规瘡涓涔夋爣绛捐繘琛屾槧灏勫苟璁℃暟
#             for original_label, mapped_label in learning_map.items():
#                 class_counts[mapped_label] += np.sum(semantic_labels == original_label)
#     return class_counts



def calculate_class_counts(yaml_path, dataset_path, ignore_label=None):
    yaml_data = load_yaml(yaml_path)
    learning_map = yaml_data['learning_map']
    train_sequences = yaml_data['split']['train']

    # 鍒濆鍖栫被鍒鏁颁负0
    max_label = max(learning_map.values())
    class_counts = np.zeros(max_label + 1, dtype=np.int64)

    for seq in train_sequences:
        seq_path = Path(dataset_path) / f"{seq:02d}" / "labels"
        for label_file in seq_path.rglob('*.label'):
            # 璇诲彇姣忎釜鐐圭殑鏍囩
            labels = np.fromfile(str(label_file), dtype=np.uint32)

            # 鑾峰彇浣?16 浣嶇殑璇箟鏍囩
            semantic_labels = labels & 0xFFFF

            # 瀵规瘡涓涔夋爣绛捐繘琛屾槧灏勫苟璁℃暟
            for original_label, mapped_label in learning_map.items():
                if original_label != ignore_label:
                    class_counts[mapped_label] += np.sum(semantic_labels == original_label)


    # 绉婚櫎蹇界暐鐨勬爣绛?    if ignore_label in learning_map:
        mapped_ignore_label = learning_map[ignore_label]
        class_counts[mapped_ignore_label] = 0    #姝ｅ父璁粌闇€瑕佹敼涓?0

    # # 濡傛灉鏈夊拷鐣ョ殑鏍囩锛岀Щ闄ゅ搴旂殑璁℃暟
    # if ignore_label is not None and ignore_label in learning_map:
    #     mapped_ignore_label = learning_map[ignore_label]
    #     class_counts = np.delete(class_counts, mapped_ignore_label)
     
    return class_counts



### 绀轰緥
# yaml_path = './config/label_mapping/semantic-kitti.yaml'  # YAML 鏂囦欢璺緞
# dataset_path = '/home/gyzhao/python-zgy/111/seq_0.2_sample'  # 鏁版嵁闆嗘爣绛剧殑璺緞

# class_counts = calculate_class_counts(yaml_path, dataset_path)
# print(class_counts)

