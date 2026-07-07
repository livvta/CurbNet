#!/usr/bin/env python3
"""
CurbNet 实时交互式推理可视化工具
===================================
基于 demo_folder_focal.py 的预处理 pipeline，实时推理并逐帧显示。
不保存文件，直接 Open3D 交互。

用法:
sh visualize_demo_interactive.sh \
    --demo-folder /home/ant/CurbNet/data/3D-Curb/07/velodyne \
    --demo-label-folder /home/ant/CurbNet/data/3D-Curb/07/labels \
    --no-input-crop \
    -y config/semantickitti-curb_0.2_12gb.yaml
    
sh visualize_demo_interactive.sh \
  --demo-folder /home/ant/CurbNet/data/NRS/transfer_velodyne \
  --demo-label-folder /home/ant/CurbNet/data/NRS/transfer_labels \
  --start-frame 380 \
  --z-shift -1.8 \
  --no-input-crop \
  -y config/semantickitti-curb_0.2_12gb.yaml
    
sh visualize_demo_interactive.sh \
    --demo-folder /home/ant/ros2_humble/dataset/industrial_bin \
    --no-input-crop \
    -y config/semantickitti-curb_0.2_12gb.yaml

键盘:
    N/D/→  下一帧（实时推理）
    P/A/←  上一帧（从缓存读取）
    L      切换预测结果/真实标签
    +/-    调整点大小
    Q/Esc  退出

终端:
    g 123  跳转到第 123 帧
    123    同上
    q      退出
"""

import os
import sys
import argparse
import time
import threading
import queue
import numpy as np
import yaml
import torch
import open3d as o3d
from tqdm import tqdm

# ---- 复用 demo_folder_focal.py 的模块 ----
from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV
from dataloader.pc_dataset import get_pc_model_class
from utils.load_save_util import load_checkpoint

import warnings
warnings.filterwarnings("ignore")


# ============ 颜色映射（复用 visualize_label_open3d.py 逻辑）============
def load_color_map(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    color_map = {}
    for label, bgr in cfg["color_map"].items():
        color_map[int(label)] = np.asarray(bgr[::-1], dtype=np.float32) / 255.0
    return color_map


def labels_to_colors(labels, color_map):
    colors = np.full((labels.shape[0], 3), 0.75, dtype=np.float32)  # 默认白灰，深色背景可见
    for label in np.unique(labels):
        colors[labels == label] = color_map.get(int(label), np.asarray([0.35, 0.35, 0.35]))
    dark_mask = np.linalg.norm(colors, axis=1) < 0.25
    colors[dark_mask] = np.asarray([0.75, 0.75, 0.75], dtype=np.float32)
    return colors


def summarize_gt_curb_region(region_mask, per_point_pred, per_point_curb_prob,
                             per_point_curb_logit, per_point_other_logit,
                             num_class, label_names):
    count = int(region_mask.sum())
    if count == 0:
        return None

    pred_counts = {}
    region_pred = per_point_pred[region_mask]
    for cls_id in range(num_class):
        cls_count = int((region_pred == cls_id).sum())
        if cls_count > 0:
            pred_counts[label_names.get(cls_id, f'class_{cls_id}')] = cls_count

    summary = {
        "count": count,
        "pred_counts": pred_counts,
    }
    if per_point_curb_prob is not None:
        summary.update({
            "curb_prob_mean": float(per_point_curb_prob[region_mask].mean()),
            "curb_prob_max": float(per_point_curb_prob[region_mask].max()),
            "curb_logit_mean": float(per_point_curb_logit[region_mask].mean()),
            "curb_margin_mean": float((per_point_curb_logit[region_mask] - per_point_other_logit[region_mask]).mean()),
            "curb_margin_max": float((per_point_curb_logit[region_mask] - per_point_other_logit[region_mask]).max()),
        })
    return summary


def format_gt_curb_summary(name, summary):
    if summary is None:
        return f"    {name:10s}: n=0"

    count = summary["count"]
    pred_counts = summary["pred_counts"]
    pred_text = ", ".join(
        f"{pred_name}:{pred_count}({pred_count / count * 100:.1f}%)"
        for pred_name, pred_count in pred_counts.items()
    )
    text = f"    {name:10s}: n={count:,} pred=[{pred_text}]"
    if "curb_prob_mean" in summary:
        text += (
            f" prob_mean={summary['curb_prob_mean']:.3e}"
            f" prob_max={summary['curb_prob_max']:.3e}"
            f" margin_mean={summary['curb_margin_mean']:.3f}"
            f" margin_max={summary['curb_margin_max']:.3f}"
        )
    return text


def build_class_stats(labels, num_class, label_names):
    total_pts = len(labels)
    stats = {}
    for cls_id in range(num_class):
        name = label_names.get(cls_id, f'class_{cls_id}')
        cnt = int((labels == cls_id).sum())
        stats[name] = (cnt, cnt / total_pts * 100 if total_pts > 0 else 0.0)
    return stats


class RuntimePreprocessDataset:
    """在内存中对 demo 点云做坐标变换、裁剪和强度归一化，不保存新文件。"""
    def __init__(self, base_dataset, min_bound, max_bound, xy_transform="none",
                 x_shift=0.0, y_shift=0.0, z_shift=0.0, intensity_scale=1.0,
                 crop_input=True, normalize_intensity=True):
        self.base_dataset = base_dataset
        self.im_idx = base_dataset.im_idx
        self.min_radius = float(min_bound[0])
        self.max_radius = float(max_bound[0])
        self.min_z = float(min_bound[2])
        self.max_z = float(max_bound[2])
        self.xy_transform = xy_transform
        self.x_shift = float(x_shift)
        self.y_shift = float(y_shift)
        self.z_shift = float(z_shift)
        self.intensity_scale = float(intensity_scale)
        self.crop_input = bool(crop_input)
        self.normalize_intensity = bool(normalize_intensity)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        data = self.base_dataset[index]
        if len(data) == 2:
            xyz, labels = data
            sig = None
        elif len(data) == 3:
            xyz, labels, sig = data
            sig = sig.copy()
        else:
            raise Exception('Return invalid data tuple')

        xyz = xyz.copy()
        x = xyz[:, 0].copy()
        y = xyz[:, 1].copy()
        if self.xy_transform == "swap":
            xyz[:, 0], xyz[:, 1] = y, x
        elif self.xy_transform == "rot90":
            xyz[:, 0], xyz[:, 1] = -y, x
        elif self.xy_transform == "rot-90":
            xyz[:, 0], xyz[:, 1] = y, -x
        elif self.xy_transform == "flip-x":
            xyz[:, 0] = -x
        elif self.xy_transform == "flip-y":
            xyz[:, 1] = -y

        if self.x_shift != 0.0:
            xyz[:, 0] += self.x_shift
        if self.y_shift != 0.0:
            xyz[:, 1] += self.y_shift
        if self.z_shift != 0.0:
            xyz[:, 2] += self.z_shift

        if self.crop_input:
            radius = np.linalg.norm(xyz[:, :2], axis=1)
            mask = (
                (radius >= self.min_radius) & (radius <= self.max_radius) &
                (xyz[:, 2] >= self.min_z) & (xyz[:, 2] <= self.max_z)
            )

            if mask.any():
                xyz = xyz[mask]
                labels = labels[mask]
                if sig is not None:
                    sig = sig[mask]

        if sig is not None:
            if self.normalize_intensity and sig.shape[0] > 0 and sig.max() > 1.0:
                sig /= 255.0
            if self.intensity_scale != 1.0:
                sig *= self.intensity_scale

        if sig is None:
            return xyz, labels
        return xyz, labels, sig


# ============ 数据集构建（从 demo_folder_focal.py 完整复用）============
def build_dataset(dataset_config, data_dir, grid_size=[480, 360, 32], demo_label_dir=None,
                  xy_transform="none", x_shift=0.0, y_shift=0.0,
                  z_shift=0.0, intensity_scale=1.0, crop_input=True,
                  normalize_intensity=True):
    if demo_label_dir == '':
        imageset = "demo"
    else:
        imageset = "val"
    label_mapping = dataset_config["label_mapping"]

    SemKITTI_demo = get_pc_model_class('SemKITTI_demo')
    demo_pt_dataset = SemKITTI_demo(data_dir, imageset=imageset,
                                    return_ref=True, label_mapping=label_mapping,
                                    demo_label_path=demo_label_dir)
    demo_pt_dataset = RuntimePreprocessDataset(
        demo_pt_dataset,
        min_bound=dataset_config['min_volume_space'],
        max_bound=dataset_config['max_volume_space'],
        xy_transform=xy_transform,
        x_shift=x_shift,
        y_shift=y_shift,
        z_shift=z_shift,
        intensity_scale=intensity_scale,
        crop_input=crop_input,
        normalize_intensity=normalize_intensity,
    )

    demo_dataset = get_model_class(dataset_config['dataset_type'])(
        demo_pt_dataset,
        grid_size=grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
    )
    demo_dataset_loader = torch.utils.data.DataLoader(
        dataset=demo_dataset,
        batch_size=1,          # 逐帧推理
        collate_fn=collate_fn_BEV,
        shuffle=False,
        num_workers=4)

    # 保留引用以便获取文件路径；demo_dataset 用于按帧随机访问推理
    return demo_dataset_loader, demo_pt_dataset, demo_dataset


# ============ 模型加载（从 demo_folder_focal.py 完整复用）============
def build_model(configs, pytorch_device):
    model_config = configs['model_params']
    train_hypers = configs['train_params']
    dataset_config = configs['dataset_params']

    model_load_path = train_hypers['model_load_path']
    if not os.path.exists(model_load_path):
        raise FileNotFoundError(f"模型不存在: {model_load_path}")

    my_model = model_builder.build(model_config)
    my_model = load_checkpoint(model_load_path, my_model)
    my_model.to(pytorch_device)
    my_model.eval()

    # loss（仅用于可选指标计算）
    yaml_path = dataset_config['label_mapping']
    train_config = configs['train_data_loader']
    train_path = train_config['data_path']
    class_counts = data_builder.calculate_class_counts(yaml_path, train_path, dataset_config['ignore_label'])
    total_samples = sum(c for c in class_counts if c > 0)
    weights = [0] * len(class_counts)
    for i, count in enumerate(class_counts):
        if count > 0:
            weights[i] = total_samples / count
    weights = torch.tensor(weights, dtype=torch.float32)
    weights /= weights.sum()
    weights = weights.to(pytorch_device)

    loss_func, lovasz_softmax = loss_builder.build(
        alpha=weights, gamma=2, wce=True, lovasz=True,
        num_class=model_config['num_class'],
        ignore_label=dataset_config['ignore_label'])

    return my_model, loss_func, lovasz_softmax, model_config, dataset_config


# ============ 单帧推理（从 demo_folder_focal.py 推理循环提取）============
def infer_single_batch(my_model, pt_fea, grid, vox_label, batch_size, pytorch_device):
    """对单个 batch 推理，返回逐点预测 (mapped label) 和 loss"""
    pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in pt_fea]
    grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in grid]
    label_tensor = vox_label.type(torch.LongTensor).to(pytorch_device)

    with torch.no_grad():
        predict_labels = my_model(pt_fea_ten, grid_ten, batch_size)

    return predict_labels, label_tensor, pt_fea, grid


# ============ 主程序 ============
def main(args):
    pytorch_device = torch.device(args.device)

    config_path = args.config_path
    configs = load_config_data(config_path)

    # --- 构建模型 ---
    my_model, loss_func, lovasz_softmax, model_config, dataset_config = \
        build_model(configs, pytorch_device)

    print(f"[INFO] 模型参数量: {sum(p.numel() for p in my_model.parameters()):,}")

    # --- 标签信息 ---
    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]
    num_class = model_config['num_class']

    # 颜色映射
    color_map = load_color_map(dataset_config["label_mapping"])
    with open(dataset_config["label_mapping"], 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    inv_learning_map = semkittiyaml['learning_map_inv']

    # --- 构建数据集 ---
    demo_dataset_loader, demo_pt_dataset, demo_dataset = build_dataset(
        dataset_config, args.demo_folder,
        grid_size=model_config['output_shape'],
        demo_label_dir=args.demo_label_folder,
        xy_transform=args.xy_transform,
        x_shift=args.x_shift,
        y_shift=args.y_shift,
        z_shift=args.z_shift,
        intensity_scale=args.intensity_scale,
        crop_input=args.input_crop,
        normalize_intensity=args.intensity_normalize)
    crop_desc = (
        f"r/z 裁剪 r=[{dataset_config['min_volume_space'][0]}, {dataset_config['max_volume_space'][0]}], "
        f"z=[{dataset_config['min_volume_space'][2]}, {dataset_config['max_volume_space'][2]}]"
        if args.input_crop else "不做 r/z 裁剪"
    )
    intensity_desc = "intensity > 1 自动除以 255" if args.intensity_normalize else "不做 intensity 归一化"
    print("[INFO] 输入预处理顺序: 坐标变换/平移 -> "
          f"{crop_desc} -> {intensity_desc}；"
          f"xy_transform={args.xy_transform}, "
          f"shift=({args.x_shift}, {args.y_shift}, {args.z_shift}), "
          f"intensity_scale={args.intensity_scale}")

    # 文件路径列表（与 DataLoader 顺序一致，shuffle=False）
    bin_paths = list(demo_pt_dataset.im_idx)
    total_frames = len(bin_paths)
    print(f"[INFO] 共 {total_frames} 帧")
    start_idx = max(0, min(args.start_frame - 1, total_frames - 1))
    if start_idx != args.start_frame - 1:
        print(f"[WARN] start-frame 超出范围，改为 Frame {start_idx + 1}")

    # --- 实时推理 + 交互式可视化 ---
    cache = {}          # frame_idx → inference/label display data
    current_idx = -1    # 当前显示的帧索引
    show_gt_labels = [False]
    has_gt_labels = args.demo_label_folder != ''
    command_queue = queue.Queue()
    command_stop = threading.Event()

    def parse_terminal_command(line):
        text = line.strip()
        if not text:
            return None

        lower = text.lower()
        if lower in ("q", "quit", "exit"):
            return ("quit", None)

        parts = text.split()
        if len(parts) == 1:
            frame_text = parts[0]
        elif len(parts) == 2 and parts[0].lower() in ("g", "go", "goto", "frame", "f"):
            frame_text = parts[1]
        else:
            print(f"[WARN] 未识别命令: {text}；输入 g 123 跳转帧，q 退出")
            return None

        try:
            frame_no = int(frame_text)
        except ValueError:
            print(f"[WARN] 非法帧号: {frame_text}")
            return None

        return ("goto", frame_no - 1)

    def terminal_command_reader():
        print("[INFO] 终端命令: 输入 g 123 跳转到第 123 帧；直接输入 123 也可以；输入 q 退出")
        while not command_stop.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:
                break
            command = parse_terminal_command(line)
            if command is not None:
                command_queue.put(command)

    # --- Open3D 初始化 ---
    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="CurbNet Real-time Inference Viewer", width=1400, height=900)
    render_opt = vis.get_render_option()
    render_opt.point_size = args.point_size
    render_opt.background_color = np.asarray([0.05, 0.05, 0.08])
    if args.show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=args.axis_size,
            origin=[0.0, 0.0, 0.0],
        )
        vis.add_geometry(axis)

    is_first_geometry = [True]  # 用列表以便在闭包中修改

    def get_or_infer(idx):
        """获取第 idx 帧的推理结果（缓存未命中则实时推理）"""
        if idx in cache:
            return cache[idx]

        try:
            _, vox_label, grid, pt_labs, pt_fea = collate_fn_BEV([demo_dataset[idx]])
        except IndexError:
            print(f"[WARN] 无法获取帧 {idx + 1}")
            return None

        batch_size_actual = vox_label.shape[0]
        predict_labels, label_tensor, _, _ = infer_single_batch(
            my_model, pt_fea, grid, vox_label, batch_size_actual, pytorch_device)

        predict_probs = torch.nn.functional.softmax(predict_labels, dim=1)
        predict_labels_np = torch.argmax(predict_labels, dim=1).cpu().numpy()
        curb_probs_np = predict_probs[:, 1].cpu().numpy() if num_class > 1 else None
        predict_logits_np = predict_labels.cpu().numpy()

        for b in range(batch_size_actual):

            # 逐点预测（通过 grid_ind 索引回原有点云）
            per_point_pred = predict_labels_np[b, grid[b][:, 0], grid[b][:, 1], grid[b][:, 2]]
            if curb_probs_np is not None:
                per_point_curb_prob = curb_probs_np[b, grid[b][:, 0], grid[b][:, 1], grid[b][:, 2]]
                per_point_curb_logit = predict_logits_np[b, 1, grid[b][:, 0], grid[b][:, 1], grid[b][:, 2]]
                per_point_other_logits = np.delete(predict_logits_np[b], 1, axis=0)
                per_point_other_logit = per_point_other_logits[:, grid[b][:, 0], grid[b][:, 1], grid[b][:, 2]].max(axis=0)
                curb_prob_stats = {
                    "mean": float(per_point_curb_prob.mean()),
                    "max": float(per_point_curb_prob.max()),
                    "min": float(per_point_curb_prob.min()),
                    "logit_mean": float(per_point_curb_logit.mean()),
                    "logit_max": float(per_point_curb_logit.max()),
                    "margin_max": float((per_point_curb_logit - per_point_other_logit).max()),
                }
            else:
                curb_prob_stats = None

            # 逆映射到原始标签
            inv_labels = np.vectorize(inv_learning_map.__getitem__)(per_point_pred).astype(np.uint32)

            # 从 pt_fea 提取 XYZ（col 5=z, 6=x, 7=y）
            # pt_fea 结构: [d_rho, d_phi, d_z, rho, phi, z, x, y, intensity]
            xyz = np.stack([
                pt_fea[b][:, 6],  # x
                pt_fea[b][:, 7],  # y
                pt_fea[b][:, 5],  # z
            ], axis=1)

            gt_curb_stats = None
            gt_labels = np.asarray(pt_labs[b]).reshape(-1)
            gt_curb_mask = gt_labels == 1  # raw label 3 is mapped to train class 1 (curb)
            if gt_curb_mask.any():
                rho = np.linalg.norm(xyz[:, :2], axis=1)
                gt_curb_stats = {
                    "all": summarize_gt_curb_region(
                        gt_curb_mask, per_point_pred, per_point_curb_prob if curb_probs_np is not None else None,
                        per_point_curb_logit if curb_probs_np is not None else None,
                        per_point_other_logit if curb_probs_np is not None else None,
                        num_class, SemKITTI_label_name),
                    f"x<{args.gt_near_x:g}": summarize_gt_curb_region(
                        gt_curb_mask & (xyz[:, 0] < args.gt_near_x),
                        per_point_pred, per_point_curb_prob if curb_probs_np is not None else None,
                        per_point_curb_logit if curb_probs_np is not None else None,
                        per_point_other_logit if curb_probs_np is not None else None,
                        num_class, SemKITTI_label_name),
                    f"x>={args.gt_near_x:g}": summarize_gt_curb_region(
                        gt_curb_mask & (xyz[:, 0] >= args.gt_near_x),
                        per_point_pred, per_point_curb_prob if curb_probs_np is not None else None,
                        per_point_curb_logit if curb_probs_np is not None else None,
                        per_point_other_logit if curb_probs_np is not None else None,
                        num_class, SemKITTI_label_name),
                    f"rho<{args.gt_near_rho:g}": summarize_gt_curb_region(
                        gt_curb_mask & (rho < args.gt_near_rho),
                        per_point_pred, per_point_curb_prob if curb_probs_np is not None else None,
                        per_point_curb_logit if curb_probs_np is not None else None,
                        per_point_other_logit if curb_probs_np is not None else None,
                        num_class, SemKITTI_label_name),
                    f"rho>={args.gt_near_rho:g}": summarize_gt_curb_region(
                        gt_curb_mask & (rho >= args.gt_near_rho),
                        per_point_pred, per_point_curb_prob if curb_probs_np is not None else None,
                        per_point_curb_logit if curb_probs_np is not None else None,
                        per_point_other_logit if curb_probs_np is not None else None,
                        num_class, SemKITTI_label_name),
                }

            pred_colors = labels_to_colors(inv_labels, color_map)
            pred_stats = build_class_stats(per_point_pred, num_class, SemKITTI_label_name)

            gt_colors = None
            gt_stats = None
            if has_gt_labels:
                inv_gt_labels = np.vectorize(inv_learning_map.__getitem__)(gt_labels).astype(np.uint32)
                gt_colors = labels_to_colors(inv_gt_labels, color_map)
                gt_stats = build_class_stats(gt_labels, num_class, SemKITTI_label_name)

            cache[idx] = {
                "xyz": xyz,
                "pred_colors": pred_colors,
                "gt_colors": gt_colors,
                "pred_stats": pred_stats,
                "gt_stats": gt_stats,
                "pred_labels_raw": inv_labels,
                "curb_prob_stats": curb_prob_stats,
                "gt_curb_stats": gt_curb_stats,
            }

        del predict_labels, vox_label, grid, pt_labs, pt_fea

        return cache.get(idx)

    def display_frame(idx, reset_view=False):
        """在 Open3D 中显示第 idx 帧"""
        nonlocal current_idx
        result = get_or_infer(idx)
        if result is None:
            return

        if isinstance(result, dict):
            xyz = result["xyz"]
            curb_prob_stats = result["curb_prob_stats"]
            gt_curb_stats = result["gt_curb_stats"]
            if show_gt_labels[0] and result["gt_colors"] is not None:
                colors = result["gt_colors"]
                stats = result["gt_stats"]
                view_name = "GT label"
            else:
                colors = result["pred_colors"]
                stats = result["pred_stats"]
                view_name = "prediction"
        elif len(result) == 4:
            xyz, colors, stats, _ = result
            curb_prob_stats = None
            gt_curb_stats = None
            view_name = "prediction"
        elif len(result) == 5:
            xyz, colors, stats, _, curb_prob_stats = result
            gt_curb_stats = None
            view_name = "prediction"
        else:
            xyz, colors, stats, _, curb_prob_stats, gt_curb_stats = result
            view_name = "prediction"
        current_idx = idx
        bin_name = os.path.basename(bin_paths[idx])

        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if is_first_geometry[0]:
            vis.add_geometry(pcd)
            is_first_geometry[0] = False
        else:
            vis.update_geometry(pcd)

        # 打印信息
        print(f"\n{'='*50}")
        print(f"  Frame {idx+1}/{total_frames}  |  {bin_name}")
        print(f"  View: {view_name}")
        print(f"  Points: {len(xyz):,}")
        if curb_prob_stats is not None:
            print("  Curb softmax: "
                  f"mean={curb_prob_stats['mean']:.3e}, "
                  f"max={curb_prob_stats['max']:.3e}, "
                  f"min={curb_prob_stats['min']:.3e}")
            print("  Curb logit: "
                  f"mean={curb_prob_stats['logit_mean']:.3f}, "
                  f"max={curb_prob_stats['logit_max']:.3f}, "
                  f"best_margin={curb_prob_stats['margin_max']:.3f}")
        if gt_curb_stats is not None:
            print("  GT curb(label=3 -> class 1):")
            for region_name, region_summary in gt_curb_stats.items():
                print(format_gt_curb_summary(region_name, region_summary))
        for name, (cnt, pct) in stats.items():
            if pct > 0.01:
                print(f"    {name:12s}: {pct:5.1f}% ({cnt:,})")
        print(f"{'='*50}")

        if reset_view:
            vis.reset_view_point(True)

    def on_next(vis_ptr):
        nonlocal current_idx
        if current_idx < total_frames - 1:
            t0 = time.time()
            display_frame(current_idx + 1)
            elapsed = time.time() - t0
            cache_hit = (current_idx) in cache and cache.get(current_idx) is not None
            # 简单判断：如果耗时 < 0.01s 则是缓存命中
            tag = "(cached)" if elapsed < 0.1 else f"({elapsed:.1f}s)"
            print(f"[NEXT] → frame {current_idx+1} {tag}")
        else:
            print("[END] 已是最后一帧")
        return True

    def on_prev(vis_ptr):
        nonlocal current_idx
        if current_idx > 0:
            t0 = time.time()
            display_frame(current_idx - 1)
            elapsed = time.time() - t0
            tag = "(cached)" if elapsed < 0.1 else f"({elapsed:.1f}s)"
            print(f"[PREV] → frame {current_idx+1} {tag}")
        else:
            print("[START] 已是第一帧")
        return True

    def on_quit(vis_ptr):
        vis.close()
        return False

    def on_size_up(vis_ptr):
        nonlocal args
        args.point_size = min(10.0, args.point_size + 0.5)
        render_opt.point_size = args.point_size
        print(f"[INFO] point_size = {args.point_size}")
        return True

    def on_size_down(vis_ptr):
        nonlocal args
        args.point_size = max(0.5, args.point_size - 0.5)
        render_opt.point_size = args.point_size
        print(f"[INFO] point_size = {args.point_size}")
        return True

    def on_reset_view(vis_ptr):
        vis.reset_view_point(True)
        print("[INFO] reset view")
        return True

    def on_toggle_labels(vis_ptr):
        if not has_gt_labels:
            print("[WARN] 当前没有真实标签；请运行时传入 --demo-label-folder")
            return True
        show_gt_labels[0] = not show_gt_labels[0]
        mode = "GT label" if show_gt_labels[0] else "prediction"
        print(f"[INFO] display mode = {mode}")
        if current_idx >= 0:
            display_frame(current_idx)
        return True

    def on_animation(vis_ptr):
        latest_goto_idx = None
        should_quit = False
        while True:
            try:
                command, value = command_queue.get_nowait()
            except queue.Empty:
                break

            if command == "quit":
                should_quit = True
            elif command == "goto":
                latest_goto_idx = value

        if latest_goto_idx is not None:
            target_idx = max(0, min(latest_goto_idx, total_frames - 1))
            if target_idx != latest_goto_idx:
                print(f"[WARN] 目标帧超出范围，改为 Frame {target_idx + 1}")
            t0 = time.time()
            display_frame(target_idx)
            elapsed = time.time() - t0
            tag = "(cached)" if elapsed < 0.1 else f"({elapsed:.1f}s)"
            print(f"[GOTO] → frame {current_idx + 1} {tag}")

        if should_quit:
            vis_ptr.close()

        return False

    # 注册键盘回调
    vis.register_key_callback(262, on_next)      # →
    vis.register_key_callback(263, on_prev)      # ←
    vis.register_key_callback(ord('N'), on_next)
    vis.register_key_callback(ord('n'), on_next)
    vis.register_key_callback(ord('D'), on_next)
    vis.register_key_callback(ord('d'), on_next)
    vis.register_key_callback(ord('P'), on_prev)
    vis.register_key_callback(ord('p'), on_prev)
    vis.register_key_callback(ord('A'), on_prev)
    vis.register_key_callback(ord('a'), on_prev)
    vis.register_key_callback(ord('Q'), on_quit)
    vis.register_key_callback(ord('q'), on_quit)
    vis.register_key_callback(256, on_quit)      # ESC
    vis.register_key_callback(ord('='), on_size_up)
    vis.register_key_callback(ord('+'), on_size_up)
    vis.register_key_callback(ord('-'), on_size_down)
    vis.register_key_callback(ord('R'), on_reset_view)
    vis.register_key_callback(ord('r'), on_reset_view)
    vis.register_key_callback(ord('L'), on_toggle_labels)
    vis.register_key_callback(ord('l'), on_toggle_labels)
    vis.register_animation_callback(on_animation)

    # --- 显示起始帧（实时推理）---
    print(f"\n[INFO] 正在推理起始帧 Frame {start_idx + 1}...")
    t0 = time.time()
    display_frame(start_idx, reset_view=True)
    print(f"[INFO] 起始帧推理耗时: {time.time()-t0:.1f}s")
    print(f"\n{'='*60}")
    print(f"  操作:  N/D/→ 下一帧  |  P/A/← 上一帧")
    print(f"        L 预测/真值切换 |  +/- 点大小")
    print(f"        R 重置视角      |  Q/Esc  退出")
    print(f"  终端:  输入 g 123 跳转到第 123 帧")
    print(f"  鼠标:  拖拽旋转 | 滚轮缩放 | 右键平移")
    print(f"{'='*60}\n")

    command_thread = threading.Thread(target=terminal_command_reader, daemon=True)
    command_thread.start()
    vis.run()
    command_stop.set()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CurbNet 实时交互式推理可视化")
    parser.add_argument('-y', '--config_path', default='config/semantickitti-curb_0.2_12gb.yaml')
    parser.add_argument('--demo-folder', type=str, required=True,
                        help='点云文件夹路径')
    parser.add_argument('--demo-label-folder', type=str, default='',
                        help='标签文件夹路径')
    parser.add_argument('--device', type=str, default='cuda:0')
    
    parser.add_argument('--point-size', type=float, default=1.5,
                        help='点云大小')
    parser.add_argument('--axis-size', type=float, default=5.0,
                        help='原点坐标系大小')
    parser.add_argument('--no-axis', dest='show_axis', action='store_false',
                        help='原点坐标系开关')
    
    parser.add_argument('--start-frame', type=int, default=1,
                        help='起始帧号')
    
    parser.add_argument('--xy-transform', default='none',
                        choices=['none', 'swap', 'rot90', 'rot-90', 'flip-x', 'flip-y'],
                        help='推理前坐标系变换 (swap:x/y 互换, rot90:逆时针转90°)')
    parser.add_argument('--x-shift', type=float, default=0.0,
                        help='推理前给 x 坐标加偏移量')
    parser.add_argument('--y-shift', type=float, default=0.0,
                        help='推理前给 y 坐标加偏移量')
    parser.add_argument('--z-shift', type=float, default=0.0,
                        help='推理前给 z 坐标加偏移量')
    
    parser.add_argument('--intensity-scale', type=float, default=1.0,
                        help='强度值再乘以该系数，用于测试强度尺度差异')
    parser.add_argument('--no-intensity-normalize', dest='intensity_normalize',
                        action='store_false',
                        help='关闭 intensity > 1 自动除以 255；默认开启')
    parser.add_argument('--no-input-crop', dest='input_crop',
                        action='store_false',
                        help='关闭运行时 r/z 裁剪')
    parser.add_argument('--gt-near-x', type=float, default=15.0,
                        help='有标签时，统计 GT curb 的近处 x 阈值 (默认 15m)')
    parser.add_argument('--gt-near-rho', type=float, default=15.0,
                        help='有标签时，统计 GT curb 的近处 rho 阈值 (默认 15m)')
    
    parser.set_defaults(intensity_normalize=True)
    parser.set_defaults(input_crop=True)
    parser.set_defaults(show_axis=True)
    args = parser.parse_args()

    if not os.path.isdir(args.demo_folder):
        print(f"[ERROR] 文件夹不存在: {args.demo_folder}")
        sys.exit(1)

    # 检查 .bin 文件
    bin_count = len([f for f in os.listdir(args.demo_folder) if f.endswith('.bin')])
    if bin_count == 0:
        print(f"[ERROR] 没有 .bin 文件: {args.demo_folder}")
        sys.exit(1)

    main(args)
