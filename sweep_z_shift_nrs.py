#!/usr/bin/env python3
"""Sweep runtime z-shift values on labeled demo scans.

This is a small evaluation helper for NRS/3D-Curb style SemanticKITTI bins.
It does not save predictions. It loads the CurbNet checkpoint once, then
rebuilds the runtime-preprocessed dataset for each z-shift and reports curb
precision/recall/F1/IoU on mapped class 1, which corresponds to raw label 3.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from builder import model_builder
from config.config import load_config_data
from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV
from dataloader.pc_dataset import get_pc_model_class
from visualize_demo_interactive import RuntimePreprocessDataset


def parse_frames(text, total_frames, start_frame, end_frame, stride):
    if text:
        frames = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                left, right = part.split("-", 1)
                begin = int(left)
                finish = int(right)
                step = 1 if finish >= begin else -1
                frames.extend(range(begin, finish + step, step))
            else:
                frames.append(int(part))
        indices = [frame_no - 1 for frame_no in frames]
    else:
        finish = total_frames if end_frame is None else min(end_frame, total_frames)
        indices = list(range(start_frame - 1, finish, stride))

    valid = sorted({idx for idx in indices if 0 <= idx < total_frames})
    if not valid:
        raise ValueError("No valid frames selected.")
    return valid


def make_z_values(args):
    if args.z_values:
        return [float(item.strip()) for item in args.z_values.split(",") if item.strip()]

    if args.z_step <= 0:
        raise ValueError("--z-step must be > 0")
    count = int(round((args.z_max - args.z_min) / args.z_step))
    values = [args.z_min + i * args.z_step for i in range(count + 1)]
    values = [value for value in values if value <= args.z_max + 1e-6]
    return [round(value, 6) for value in values]


def build_model(configs, device):
    model_config = configs["model_params"]
    model_load_path = configs["train_params"]["model_load_path"]
    if not os.path.exists(model_load_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_load_path}")

    model = model_builder.build(model_config)
    model = load_checkpoint_for_device(model_load_path, model, device)
    model.to(device)
    model.eval()
    return model, model_config


def load_checkpoint_for_device(model_load_path, model, device):
    model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path, map_location=device)

    part_load = {}
    match_size = 0
    nomatch_size = 0
    for key, value in pre_weight.items():
        if key in model_dict and model_dict[key].shape == value.shape:
            match_size += 1
            part_load[key] = value
        else:
            nomatch_size += 1

    print(f"matched parameter sets: {match_size}, and no matched: {nomatch_size}")
    model_dict.update(part_load)
    model.load_state_dict(model_dict)
    return model


def build_eval_dataset(dataset_config, model_config, args, z_shift):
    SemKITTI_demo = get_pc_model_class("SemKITTI_demo")
    pt_dataset = SemKITTI_demo(
        args.demo_folder,
        imageset="val",
        return_ref=True,
        label_mapping=dataset_config["label_mapping"],
        demo_label_path=args.demo_label_folder,
    )
    pt_dataset = RuntimePreprocessDataset(
        pt_dataset,
        min_bound=dataset_config["min_volume_space"],
        max_bound=dataset_config["max_volume_space"],
        xy_transform=args.xy_transform,
        x_shift=args.x_shift,
        y_shift=args.y_shift,
        z_shift=z_shift,
        intensity_scale=args.intensity_scale,
        crop_input=args.input_crop,
        normalize_intensity=args.intensity_normalize,
    )
    dataset = get_model_class(dataset_config["dataset_type"])(
        pt_dataset,
        grid_size=model_config["output_shape"],
        fixed_volume_space=dataset_config["fixed_volume_space"],
        max_volume_space=dataset_config["max_volume_space"],
        min_volume_space=dataset_config["min_volume_space"],
        ignore_label=dataset_config["ignore_label"],
    )
    return dataset, pt_dataset


def empty_counts():
    return {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0, "points": 0}


def update_counts(counts, pred, gt, region):
    pred = pred[region]
    gt = gt[region]
    pred_curb = pred == 1
    gt_curb = gt == 1
    counts["tp"] += int((pred_curb & gt_curb).sum())
    counts["fp"] += int((pred_curb & ~gt_curb).sum())
    counts["fn"] += int((~pred_curb & gt_curb).sum())
    counts["gt"] += int(gt_curb.sum())
    counts["pred"] += int(pred_curb.sum())
    counts["points"] += int(region.sum())


def finalize_counts(counts):
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    result = dict(counts)
    result.update({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    })
    return result


def infer_frame(model, dataset, idx, device):
    _, vox_label, grid, pt_labs, pt_fea = collate_fn_BEV([dataset[idx]])
    pt_fea_ten = [torch.from_numpy(item).type(torch.FloatTensor).to(device) for item in pt_fea]
    grid_ten = [torch.from_numpy(item).to(device) for item in grid]

    with torch.no_grad():
        logits = model(pt_fea_ten, grid_ten, vox_label.shape[0])
        pred_volume = torch.argmax(logits, dim=1).cpu().numpy()

    grid_idx = grid[0]
    per_point_pred = pred_volume[0, grid_idx[:, 0], grid_idx[:, 1], grid_idx[:, 2]]
    per_point_gt = np.asarray(pt_labs[0]).reshape(-1)
    xyz = np.stack([pt_fea[0][:, 6], pt_fea[0][:, 7], pt_fea[0][:, 5]], axis=1)
    return per_point_pred, per_point_gt, xyz


def evaluate_shift(model, dataset, frame_indices, device, near_x, near_rho):
    counts = {
        "all": empty_counts(),
        "x_near": empty_counts(),
        "x_far": empty_counts(),
        "rho_near": empty_counts(),
        "rho_far": empty_counts(),
    }

    for idx in tqdm(frame_indices, desc="frames", leave=False):
        pred, gt, xyz = infer_frame(model, dataset, idx, device)
        rho = np.linalg.norm(xyz[:, :2], axis=1)
        all_region = np.ones(gt.shape[0], dtype=bool)

        update_counts(counts["all"], pred, gt, all_region)
        update_counts(counts["x_near"], pred, gt, xyz[:, 0] < near_x)
        update_counts(counts["x_far"], pred, gt, xyz[:, 0] >= near_x)
        update_counts(counts["rho_near"], pred, gt, rho < near_rho)
        update_counts(counts["rho_far"], pred, gt, rho >= near_rho)

    return {name: finalize_counts(value) for name, value in counts.items()}


def print_row(row):
    print(
        f"z={row['z_shift']:6.2f} | "
        f"all F1={row['all_f1']*100:6.2f} IoU={row['all_iou']*100:6.2f} "
        f"P={row['all_precision']*100:6.2f} R={row['all_recall']*100:6.2f} | "
        f"x<{row['near_x']:g} F1={row['x_near_f1']*100:6.2f} R={row['x_near_recall']*100:6.2f} | "
        f"x>={row['near_x']:g} F1={row['x_far_f1']*100:6.2f} R={row['x_far_recall']*100:6.2f}"
    )


def flatten_result(z_shift, metrics, near_x, near_rho, seconds):
    row = {
        "z_shift": z_shift,
        "near_x": near_x,
        "near_rho": near_rho,
        "seconds": seconds,
    }
    for region, values in metrics.items():
        for key, value in values.items():
            row[f"{region}_{key}"] = value
    return row


def main(args):
    if not os.path.isdir(args.demo_folder):
        raise FileNotFoundError(args.demo_folder)
    if not os.path.isdir(args.demo_label_folder):
        raise FileNotFoundError(args.demo_label_folder)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"{args.device} requested but CUDA is not available. "
            "This CurbNet/spconv inference path needs a CUDA runtime."
        )
    device = torch.device(args.device)
    configs = load_config_data(args.config_path)
    dataset_config = configs["dataset_params"]
    model, model_config = build_model(configs, device)

    z_values = make_z_values(args)

    first_dataset, first_pt_dataset = build_eval_dataset(dataset_config, model_config, args, z_values[0])
    frame_indices = parse_frames(args.frames, len(first_pt_dataset), args.start_frame, args.end_frame, args.stride)

    print(f"[INFO] frames: {len(frame_indices)} selected from {len(first_pt_dataset)} total")
    print(f"[INFO] first/last selected frame: {frame_indices[0] + 1}/{frame_indices[-1] + 1}")
    print(f"[INFO] z shifts: {', '.join(f'{z:.2f}' for z in z_values)}")
    print(f"[INFO] metrics: raw label=3 mapped to class 1 curb, argmax prediction")

    rows = []
    for z_shift in z_values:
        t0 = time.time()
        if z_shift == z_values[0]:
            dataset = first_dataset
        else:
            dataset, _ = build_eval_dataset(dataset_config, model_config, args, z_shift)
        metrics = evaluate_shift(model, dataset, frame_indices, device, args.near_x, args.near_rho)
        seconds = time.time() - t0
        row = flatten_result(z_shift, metrics, args.near_x, args.near_rho, seconds)
        rows.append(row)
        print_row(row)

    best_all = max(rows, key=lambda item: item["all_f1"])
    best_near = max(rows, key=lambda item: item["x_near_f1"])
    best_balanced = max(rows, key=lambda item: 0.5 * item["all_f1"] + 0.5 * item["x_near_f1"])

    print("\n[RESULT]")
    print(f"best all-curb F1      : z={best_all['z_shift']:.2f}, F1={best_all['all_f1']*100:.2f}, IoU={best_all['all_iou']*100:.2f}")
    print(f"best x<{args.near_x:g} curb F1 : z={best_near['z_shift']:.2f}, F1={best_near['x_near_f1']*100:.2f}, IoU={best_near['x_near_iou']*100:.2f}")
    print(f"best balanced F1      : z={best_balanced['z_shift']:.2f}, score={(0.5 * best_balanced['all_f1'] + 0.5 * best_balanced['x_near_f1'])*100:.2f}")

    if args.csv:
        fieldnames = sorted(rows[0].keys())
        with open(args.csv, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] wrote CSV: {args.csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep z-shift values on labeled NRS frames.")
    parser.add_argument("-y", "--config_path", default="config/semantickitti-curb_0.2_12gb.yaml")
    parser.add_argument("--demo-folder", required=True, help="Folder containing .bin point clouds")
    parser.add_argument("--demo-label-folder", required=True, help="Folder containing .label files")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--z-min", type=float, default=-2.1)
    parser.add_argument("--z-max", type=float, default=-1.1)
    parser.add_argument("--z-step", type=float, default=0.1)
    parser.add_argument("--z-values", default="", help="Comma-separated z values, e.g. -1.5,-1.6,-1.7")
    parser.add_argument("--frames", default="", help="1-based frames/ranges, e.g. 382-386,1000,1200-1210")
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--near-x", type=float, default=15.0)
    parser.add_argument("--near-rho", type=float, default=15.0)
    parser.add_argument("--xy-transform", default="none",
                        choices=["none", "swap", "rot90", "rot-90", "flip-x", "flip-y"])
    parser.add_argument("--x-shift", type=float, default=0.0)
    parser.add_argument("--y-shift", type=float, default=0.0)
    parser.add_argument("--intensity-scale", type=float, default=1.0)
    parser.add_argument("--no-intensity-normalize", dest="intensity_normalize", action="store_false",
                        help="Disable intensity > 1 automatic divide-by-255 normalization")
    parser.add_argument("--no-input-crop", dest="input_crop", action="store_false",
                        help="Disable runtime r/z crop")
    parser.add_argument("--csv", default="", help="Optional CSV output path")
    parser.set_defaults(intensity_normalize=True)
    parser.set_defaults(input_crop=True)
    main(parser.parse_args())
