
"""
此代码用于 focal loss版本的 CurbNet模型 推理测试

Attention：如果使用不一样的模型结构版本进行测试，务必要修改model_builder里面调用的model structure
"""

import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import yaml

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV
from dataloader.pc_dataset import get_pc_model_class

from utils.load_save_util import load_checkpoint

import warnings


warnings.filterwarnings("ignore")


def build_dataset(dataset_config,
                  data_dir,
                  grid_size=[480, 360, 32],
                  demo_label_dir=None):

    if demo_label_dir == '':
        imageset = "demo"
    else:
        imageset = "val"
    label_mapping = dataset_config["label_mapping"]

    SemKITTI_demo = get_pc_model_class('SemKITTI_demo')

    demo_pt_dataset = SemKITTI_demo(data_dir, imageset=imageset,
                              return_ref=True, label_mapping=label_mapping, demo_label_path=demo_label_dir)

    demo_dataset = get_model_class(dataset_config['dataset_type'])(
        demo_pt_dataset,
        grid_size=grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
    )
    demo_dataset_loader = torch.utils.data.DataLoader(dataset=demo_dataset,
                                                     batch_size=10,
                                                     collate_fn=collate_fn_BEV,
                                                     shuffle=False,
                                                     num_workers=12)  # batch_size=1  num_workers=4

    return demo_dataset_loader

def main(args):

    start_time = time.time()  # 记录开始时间  TIME  #############


    pytorch_device = torch.device('cuda:0') ###########################################  设置cuda
    # torch.cuda.empty_cache()

    config_path = args.config_path
    configs = load_config_data(config_path)
    dataset_config = configs['dataset_params']
    data_dir = args.demo_folder
    demo_label_dir = args.demo_label_folder
    save_dir = args.save_folder + "/"

    demo_batch_size = 10  # 1
    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']
    model_load_path = train_hypers['model_load_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

##############################################################
    yaml_path = dataset_config['label_mapping']
    train_config = configs['train_data_loader']
    train_path = train_config['data_path']
    # print('yaml_path:', yaml_path)
    # print('train_path:', train_path)
    # print('unique_label_str:', unique_label_str)

    class_counts = data_builder.calculate_class_counts(yaml_path, train_path, ignore_label)
    print('class_counts:', class_counts)
    # 计算总样本数（排除计数为0的类别）
    total_samples = sum(c for c in class_counts if c > 0)
    # 初始化一个与class_counts相同长度的列表，用于存储权重
    weights = [0] * len(class_counts)
    # 计算每个非零类别的权重
    for i, count in enumerate(class_counts):
        if count > 0:
            weights[i] = total_samples / count   #常见的方法是使权重与类别频率成反比。用总样本数量除以每个类别的样本数量，然后对这些比例进行归一化，以便它们加起来等于类别数
    # 将weights转换为Tensor并归一化
    weights = torch.tensor(weights, dtype=torch.float32)
    weights /= weights.sum()
    # 将权重转移到PyTorch设备上
    weights = weights.to(pytorch_device)
    print('weights:', weights)
##############################################################

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    # loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
    #                                                num_class=num_class, ignore_label=ignore_label)
    loss_func, lovasz_softmax = loss_builder.build(alpha=weights, gamma=2, wce=True, lovasz=True,
                                               num_class=num_class, ignore_label=ignore_label)

    demo_dataset_loader = build_dataset(dataset_config, data_dir, grid_size=grid_size, demo_label_dir=demo_label_dir)
    with open(dataset_config["label_mapping"], 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    inv_learning_map = semkittiyaml['learning_map_inv']

    my_model.eval()
    hist_list = []
    demo_loss_list = []

    inference_start_time = time.time()  # 记录推理开始时间 TIME   ##############

    with torch.no_grad():
        for i_iter_demo, (_, demo_vox_label, demo_grid, demo_pt_labs, demo_pt_fea) in enumerate(
                demo_dataset_loader):

            inference_start_time1 = time.time()  # 记录推理开始时间 TIME   ##############

            demo_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                              demo_pt_fea]
            demo_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in demo_grid]
            demo_label_tensor = demo_vox_label.type(torch.LongTensor).to(pytorch_device)

            predict_labels = my_model(demo_pt_fea_ten, demo_grid_ten, demo_batch_size)
            loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), demo_label_tensor,
                                  ignore=0) + loss_func(predict_labels.detach(), demo_label_tensor)
            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()
            for count, i_demo_grid in enumerate(demo_grid):
                hist_list.append(fast_hist_crop(predict_labels[
                                                    count, demo_grid[count][:, 0], demo_grid[count][:, 1],
                                                    demo_grid[count][:, 2]], demo_pt_labs[count],
                                                unique_label))
                inv_labels = np.vectorize(inv_learning_map.__getitem__)(predict_labels[count, demo_grid[count][:, 0], demo_grid[count][:, 1], demo_grid[count][:, 2]]) 
                inv_labels = inv_labels.astype('uint32')
                outputPath = save_dir + str(i_iter_demo).zfill(6) + '.label'
                inv_labels.tofile(outputPath)
                print("save " + outputPath)

            inference_end_time1 = time.time()  # 记录推理结束时间  TIME  #############
            inference_duration1 = inference_end_time1 - inference_start_time1  # 计算推理时间 TIME   #############
            print("AAAAAA", inference_duration1)

            demo_loss_list.append(loss.detach().cpu().numpy())

    inference_end_time = time.time()  # 记录推理结束时间  TIME  #############
    inference_duration = inference_end_time - inference_start_time  # 计算推理时间 TIME   #############
    print("BBBBB", inference_duration)

    if demo_label_dir != '':
        my_model.train()
        iou = per_class_iu(sum(hist_list))
        print('Validation per class iou: ')
        for class_name, class_iou in zip(unique_label_str, iou):
            print('%s : %.2f%%' % (class_name, class_iou * 100))
        val_miou = np.nanmean(iou) * 100
        del demo_vox_label, demo_grid, demo_pt_fea, demo_grid_ten

        print('Current val miou is %.3f' %
              (val_miou))
        print('Current val loss is %.3f' %
              (np.mean(demo_loss_list)))

############ TIME
    end_time = time.time()  # 记录结束时间
    total_duration = end_time - start_time  # 计算总时间

    # print('Inference duration: %.2f seconds' % inference_duration)
    # print('Total duration: %.2f seconds' % total_duration)
############


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti-curb_0.2.yaml')    # config path of model and dataset
    parser.add_argument('--demo-folder', type=str, default='', help='path to the folder containing demo lidar scans', required=True)
    parser.add_argument('--save-folder', type=str, default='', help='path to save your result', required=True)
    parser.add_argument('--demo-label-folder', type=str, default='', help='path to the folder containing demo labels')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)

