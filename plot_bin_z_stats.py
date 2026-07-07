#!/usr/bin/env python3
"""Plot z-axis statistics for one or more SemanticKITTI-style .bin files.

# 单个文件
/home/ant/miniconda3/envs/curbnet/bin/python plot_bin_z_stats.py \
  /home/ant/CurbNet/data/3D-Curb/00/velodyne/000008.bin \
  --label-folder /home/ant/CurbNet/data/3D-Curb/00/labels

# 多个文件
/home/ant/miniconda3/envs/curbnet/bin/python plot_bin_z_stats.py \
  /home/ant/CurbNet/data/NRS/transfer_velodyne/20220705102852_Sunny_City_Day_0163.bin \
  /home/ant/CurbNet/data/NRS/transfer_velodyne/20220705102852_Sunny_City_Day_2224.bin \
  --label-folder /home/ant/CurbNet/data/NRS/transfer_labels


# 目录或 glob
/home/ant/miniconda3/envs/curbnet/bin/python plot_bin_z_stats.py \
  '/home/ant/CurbNet/data/NRS/transfer_velodyne/*.bin' \
  -o /tmp/nrs_z_distribution.svg

"""




import argparse
import glob
from pathlib import Path

import numpy as np


DEFAULT_CLASS_COLORS = {
    "curb": "#18b83f",
    "crub": "#18b83f",
    "road": "#d21fd2",
}


def collect_bin_files(inputs):
    files = []
    for item in inputs:
        matches = glob.glob(item)
        if matches:
            candidates = matches
        else:
            candidates = [item]

        for candidate in candidates:
            path = Path(candidate)
            if path.is_dir():
                files.extend(sorted(path.glob("*.bin")))
            elif path.is_file() and path.suffix == ".bin":
                files.append(path)

    unique = []
    seen = set()
    for path in files:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def load_z_values(bin_files):
    z_values = []
    point_counts = []
    for path in bin_files:
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4 != 0:
            raise ValueError(f"{path} float32 count {raw.size} cannot reshape to Nx4")
        pts = raw.reshape(-1, 4)
        z_values.append(pts[:, 2])
        point_counts.append(pts.shape[0])
    return np.concatenate(z_values), point_counts


def parse_label_values(text):
    classes = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, value = item.split(":", 1)
            classes.append((name.strip(), int(value.strip())))
        else:
            value = int(item)
            classes.append((f"label_{value}", value))
    if not classes:
        raise ValueError("No valid --label-values entries")
    return classes


def label_path_for_bin(bin_path, label_folder):
    label_path = Path(label_folder) / f"{bin_path.stem}.label"
    if not label_path.is_file():
        raise FileNotFoundError(f"Label not found for {bin_path.name}: {label_path}")
    return label_path


def load_labeled_z_values(bin_files, label_folder, classes):
    grouped_z = {name: [] for name, _ in classes}
    point_counts = []
    label_counts = {name: 0 for name, _ in classes}

    for bin_path in bin_files:
        raw = np.fromfile(bin_path, dtype=np.float32)
        if raw.size % 4 != 0:
            raise ValueError(f"{bin_path} float32 count {raw.size} cannot reshape to Nx4")
        pts = raw.reshape(-1, 4)

        labels = np.fromfile(label_path_for_bin(bin_path, label_folder), dtype=np.uint32)
        labels = labels & 0xFFFF
        if labels.shape[0] != pts.shape[0]:
            raise ValueError(
                f"Point/label count mismatch for {bin_path.name}: "
                f"{pts.shape[0]} points vs {labels.shape[0]} labels"
            )

        point_counts.append(pts.shape[0])
        for name, label_value in classes:
            mask = labels == label_value
            label_counts[name] += int(mask.sum())
            if mask.any():
                grouped_z[name].append(pts[mask, 2])

    grouped_z = {
        name: np.concatenate(values) if values else np.empty((0,), dtype=np.float32)
        for name, values in grouped_z.items()
    }
    return grouped_z, point_counts, label_counts


def smooth_counts(counts, window):
    if window <= 1:
        return counts.astype(np.float64)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(counts.astype(np.float64), kernel, mode="same")


def find_peaks(bin_centers, counts, max_peaks, min_prominence_ratio, min_peak_distance):
    if counts.size < 3:
        return []

    max_count = counts.max()
    if max_count <= 0:
        return []

    min_prominence = max_count * min_prominence_ratio
    peak_indices = []
    for idx in range(1, counts.size - 1):
        if counts[idx] <= counts[idx - 1] or counts[idx] <= counts[idx + 1]:
            continue
        left_min = counts[:idx + 1].min()
        right_min = counts[idx:].min()
        prominence = counts[idx] - max(left_min, right_min)
        if prominence >= min_prominence:
            peak_indices.append(idx)

    peak_indices.sort(key=lambda i: counts[i], reverse=True)
    filtered = []
    for idx in peak_indices:
        if all(abs(bin_centers[idx] - bin_centers[kept]) >= min_peak_distance for kept in filtered):
            filtered.append(idx)
        if len(filtered) >= max_peaks:
            break

    peak_indices = filtered
    peak_indices.sort(key=lambda i: bin_centers[i])
    return [(float(bin_centers[i]), float(counts[i]), int(i)) for i in peak_indices]


def print_summary(z, point_counts, bin_files, peaks):
    q = np.percentile(z, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print("[INFO] files:", len(bin_files))
    print("[INFO] points:", f"{z.size:,}")
    print("[INFO] per-file points: min={}, median={}, max={}".format(
        f"{int(np.min(point_counts)):,}",
        f"{int(np.median(point_counts)):,}",
        f"{int(np.max(point_counts)):,}",
    ))
    print("[INFO] z percentiles:")
    for name, value in zip(["min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"], q):
        print(f"  {name:>3s}: {value:8.3f}")

    if peaks:
        print("[INFO] peaks:")
        for rank, (z_peak, count, _) in enumerate(peaks, start=1):
            print(f"  #{rank}: z={z_peak:.3f}, count={count:.0f}")
    else:
        print("[INFO] peaks: none detected")


def print_group_summary(grouped_z, point_counts, bin_files, group_peaks):
    print("[INFO] files:", len(bin_files))
    print("[INFO] total points in files:", f"{int(np.sum(point_counts)):,}")
    print("[INFO] per-file points: min={}, median={}, max={}".format(
        f"{int(np.min(point_counts)):,}",
        f"{int(np.median(point_counts)):,}",
        f"{int(np.max(point_counts)):,}",
    ))

    for name, z in grouped_z.items():
        print(f"[INFO] class {name}: points={z.size:,}")
        if z.size == 0:
            print("  no points")
            continue
        q = np.percentile(z, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        for q_name, value in zip(["min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"], q):
            print(f"  {q_name:>3s}: {value:8.3f}")
        peaks = group_peaks.get(name, [])
        if peaks:
            print("  peaks:")
            for rank, (z_peak, count, _) in enumerate(peaks, start=1):
                print(f"    #{rank}: z={z_peak:.3f}, count={count:.0f}")
        else:
            print("  peaks: none detected")


def plot_histogram_matplotlib(z, bin_edges, counts, smoothed, peaks, args, bin_files):
    import matplotlib.pyplot as plt

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    fig, ax = plt.subplots(figsize=(12, 6), dpi=args.dpi)

    width = np.diff(bin_edges)
    ax.bar(bin_centers, counts, width=width, align="center",
           color="#8fb9d9", edgecolor="#4f6f85", linewidth=0.25, alpha=0.75,
           label="histogram")
    if args.smooth_window > 1:
        ax.plot(bin_centers, smoothed, color="#c23b22", linewidth=1.8,
                label=f"smoothed(window={args.smooth_window})")

    for rank, (z_peak, count, idx) in enumerate(peaks, start=1):
        y_value = smoothed[idx] if args.smooth_window > 1 else counts[idx]
        ax.axvline(z_peak, color="#111111", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.scatter([z_peak], [y_value], color="#111111", s=28, zorder=4)
        ax.annotate(
            f"#{rank} z={z_peak:.2f}",
            xy=(z_peak, y_value),
            xytext=(6, 10 + 12 * ((rank - 1) % 3)),
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "#111111", "linewidth": 0.8},
        )

    title_name = bin_files[0].name if len(bin_files) == 1 else f"{len(bin_files)} bin files"
    ax.set_title(f"Z Distribution - {title_name}")
    ax.set_xlabel("z (m)")
    ax.set_ylabel("point count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)
    return args.output


def plot_histogram_svg(z, bin_edges, counts, smoothed, peaks, args, bin_files):
    output = args.output
    if not output.lower().endswith(".svg"):
        output = str(Path(output).with_suffix(".svg"))
        print(f"[WARN] matplotlib not installed; writing SVG instead: {output}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    width = 1200
    height = 640
    margin_left = 78
    margin_right = 28
    margin_top = 58
    margin_bottom = 78
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    x_min = float(bin_edges[0])
    x_max = float(bin_edges[-1])
    y_max = float(max(counts.max(), smoothed.max(), 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5

    def sx(x):
        return margin_left + (float(x) - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return margin_top + plot_h - float(y) / y_max * plot_h

    title_name = bin_files[0].name if len(bin_files) == 1 else f"{len(bin_files)} bin files"
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="20">Z Distribution - {title_name}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#222"/>',
    ]

    for frac in np.linspace(0, 1, 6):
        y_val = y_max * frac
        y_pos = sy(y_val)
        elements.append(f'<line x1="{margin_left}" y1="{y_pos:.1f}" x2="{margin_left + plot_w}" y2="{y_pos:.1f}" stroke="#ddd"/>')
        elements.append(f'<text x="{margin_left - 8}" y="{y_pos + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{y_val:.0f}</text>')

    for x_val in np.linspace(x_min, x_max, 8):
        x_pos = sx(x_val)
        elements.append(f'<line x1="{x_pos:.1f}" y1="{margin_top + plot_h}" x2="{x_pos:.1f}" y2="{margin_top + plot_h + 5}" stroke="#222"/>')
        elements.append(f'<text x="{x_pos:.1f}" y="{margin_top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="11">{x_val:.2f}</text>')

    bar_pad = 0.5
    for left, right, count in zip(bin_edges[:-1], bin_edges[1:], counts):
        x1 = sx(left) + bar_pad
        x2 = sx(right) - bar_pad
        y1 = sy(count)
        y2 = margin_top + plot_h
        if x2 > x1:
            elements.append(f'<rect x="{x1:.2f}" y="{y1:.2f}" width="{x2 - x1:.2f}" height="{y2 - y1:.2f}" fill="#8fb9d9" opacity="0.78"/>')

    if args.smooth_window > 1:
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(bin_centers, smoothed))
        elements.append(f'<polyline points="{points}" fill="none" stroke="#c23b22" stroke-width="2"/>')

    for rank, (z_peak, _, idx) in enumerate(peaks, start=1):
        y_value = smoothed[idx] if args.smooth_window > 1 else counts[idx]
        x_pos = sx(z_peak)
        y_pos = sy(y_value)
        label_y = max(margin_top + 14, y_pos - 12 - 16 * ((rank - 1) % 3))
        elements.extend([
            f'<line x1="{x_pos:.2f}" y1="{margin_top}" x2="{x_pos:.2f}" y2="{margin_top + plot_h}" stroke="#111" stroke-dasharray="5,4" opacity="0.65"/>',
            f'<circle cx="{x_pos:.2f}" cy="{y_pos:.2f}" r="4" fill="#111"/>',
            f'<text x="{x_pos + 6:.2f}" y="{label_y:.2f}" font-family="Arial" font-size="12">#{rank} z={z_peak:.2f}</text>',
        ])

    elements.extend([
        f'<text x="{margin_left + plot_w / 2}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="14">z (m)</text>',
        f'<text x="22" y="{margin_top + plot_h / 2}" transform="rotate(-90 22 {margin_top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="14">point count</text>',
        "</svg>",
    ])

    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(elements))
    return output


def plot_histogram(z, bin_edges, counts, smoothed, peaks, args, bin_files):
    try:
        return plot_histogram_matplotlib(z, bin_edges, counts, smoothed, peaks, args, bin_files)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        return plot_histogram_svg(z, bin_edges, counts, smoothed, peaks, args, bin_files)


def class_color(name, index):
    fallback = ["#18b83f", "#d21fd2", "#2468d8", "#d89024", "#111111"]
    return DEFAULT_CLASS_COLORS.get(name.lower(), fallback[index % len(fallback)])


def plot_group_histogram_matplotlib(bin_edges, group_counts, group_smoothed, group_peaks, args, bin_files):
    import matplotlib.pyplot as plt

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    width = np.diff(bin_edges)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=args.dpi)

    for idx, (name, counts) in enumerate(group_counts.items()):
        color = class_color(name, idx)
        ax.bar(bin_centers, counts, width=width, align="center",
               color=color, linewidth=0, alpha=0.25, label=f"{name} histogram")
        if args.smooth_window > 1:
            ax.plot(bin_centers, group_smoothed[name], color=color, linewidth=2.0,
                    label=f"{name} smoothed")

        for rank, (z_peak, _, peak_idx) in enumerate(group_peaks.get(name, []), start=1):
            y_value = group_smoothed[name][peak_idx] if args.smooth_window > 1 else counts[peak_idx]
            ax.axvline(z_peak, color=color, linestyle="--", linewidth=1.0, alpha=0.65)
            ax.scatter([z_peak], [y_value], color=color, s=28, zorder=4)
            ax.annotate(
                f"{name}#{rank} z={z_peak:.2f}",
                xy=(z_peak, y_value),
                xytext=(6, 10 + 12 * ((rank - 1 + idx) % 4)),
                textcoords="offset points",
                fontsize=9,
                color=color,
                arrowprops={"arrowstyle": "->", "color": color, "linewidth": 0.8},
            )

    title_name = bin_files[0].name if len(bin_files) == 1 else f"{len(bin_files)} bin files"
    ax.set_title(f"Road/Curb Z Distribution - {title_name}")
    ax.set_xlabel("z (m)")
    ax.set_ylabel("point count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)
    return args.output


def plot_group_histogram_svg(bin_edges, group_counts, group_smoothed, group_peaks, args, bin_files):
    output = args.output
    if not output.lower().endswith(".svg"):
        output = str(Path(output).with_suffix(".svg"))
        print(f"[WARN] matplotlib not installed; writing SVG instead: {output}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    width = 1200
    height = 680
    margin_left = 78
    margin_right = 180
    margin_top = 58
    margin_bottom = 78
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    x_min = float(bin_edges[0])
    x_max = float(bin_edges[-1])
    y_max = max(
        [counts.max() for counts in group_counts.values() if counts.size] +
        [smoothed.max() for smoothed in group_smoothed.values() if smoothed.size] +
        [1.0]
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5

    def sx(x):
        return margin_left + (float(x) - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return margin_top + plot_h - float(y) / y_max * plot_h

    title_name = bin_files[0].name if len(bin_files) == 1 else f"{len(bin_files)} bin files"
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="20">Road/Curb Z Distribution - {title_name}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#222"/>',
    ]

    for frac in np.linspace(0, 1, 6):
        y_val = y_max * frac
        y_pos = sy(y_val)
        elements.append(f'<line x1="{margin_left}" y1="{y_pos:.1f}" x2="{margin_left + plot_w}" y2="{y_pos:.1f}" stroke="#ddd"/>')
        elements.append(f'<text x="{margin_left - 8}" y="{y_pos + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{y_val:.0f}</text>')

    for x_val in np.linspace(x_min, x_max, 8):
        x_pos = sx(x_val)
        elements.append(f'<line x1="{x_pos:.1f}" y1="{margin_top + plot_h}" x2="{x_pos:.1f}" y2="{margin_top + plot_h + 5}" stroke="#222"/>')
        elements.append(f'<text x="{x_pos:.1f}" y="{margin_top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="11">{x_val:.2f}</text>')

    group_items = list(group_counts.items())
    for group_idx, (name, counts) in enumerate(group_items):
        color = class_color(name, group_idx)
        offset_frac = (group_idx - (len(group_items) - 1) / 2.0) * 0.18
        for left, right, count in zip(bin_edges[:-1], bin_edges[1:], counts):
            bin_w = sx(right) - sx(left)
            x1 = sx(left) + bin_w * (0.12 + offset_frac)
            x2 = sx(right) - bin_w * (0.12 - offset_frac)
            y1 = sy(count)
            y2 = margin_top + plot_h
            if x2 > x1:
                elements.append(f'<rect x="{x1:.2f}" y="{y1:.2f}" width="{x2 - x1:.2f}" height="{y2 - y1:.2f}" fill="{color}" opacity="0.28"/>')

        if args.smooth_window > 1:
            points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(bin_centers, group_smoothed[name]))
            elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>')

        for rank, (z_peak, _, peak_idx) in enumerate(group_peaks.get(name, []), start=1):
            y_value = group_smoothed[name][peak_idx] if args.smooth_window > 1 else counts[peak_idx]
            x_pos = sx(z_peak)
            y_pos = sy(y_value)
            label_y = max(margin_top + 14, y_pos - 12 - 16 * ((rank - 1 + group_idx) % 4))
            elements.extend([
                f'<line x1="{x_pos:.2f}" y1="{margin_top}" x2="{x_pos:.2f}" y2="{margin_top + plot_h}" stroke="{color}" stroke-dasharray="5,4" opacity="0.65"/>',
                f'<circle cx="{x_pos:.2f}" cy="{y_pos:.2f}" r="4" fill="{color}"/>',
                f'<text x="{x_pos + 6:.2f}" y="{label_y:.2f}" font-family="Arial" font-size="12" fill="{color}">{name}#{rank} z={z_peak:.2f}</text>',
            ])

    legend_x = margin_left + plot_w + 24
    legend_y = margin_top + 8
    for group_idx, (name, _) in enumerate(group_items):
        color = class_color(name, group_idx)
        y = legend_y + group_idx * 24
        elements.append(f'<rect x="{legend_x}" y="{y - 10}" width="14" height="14" fill="{color}" opacity="0.55"/>')
        elements.append(f'<text x="{legend_x + 22}" y="{y + 2}" font-family="Arial" font-size="13">{name}</text>')

    elements.extend([
        f'<text x="{margin_left + plot_w / 2}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="14">z (m)</text>',
        f'<text x="22" y="{margin_top + plot_h / 2}" transform="rotate(-90 22 {margin_top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="14">point count</text>',
        "</svg>",
    ])

    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(elements))
    return output


def plot_group_histogram(bin_edges, group_counts, group_smoothed, group_peaks, args, bin_files):
    try:
        return plot_group_histogram_matplotlib(bin_edges, group_counts, group_smoothed, group_peaks, args, bin_files)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        return plot_group_histogram_svg(bin_edges, group_counts, group_smoothed, group_peaks, args, bin_files)


def main(args):
    bin_files = collect_bin_files(args.inputs)
    if not bin_files:
        raise FileNotFoundError("No .bin files found from inputs")

    if args.label_folder:
        classes = parse_label_values(args.label_values)
        grouped_z, point_counts, _ = load_labeled_z_values(bin_files, args.label_folder, classes)

        if args.z_min is not None or args.z_max is not None:
            z_min = -np.inf if args.z_min is None else args.z_min
            z_max = np.inf if args.z_max is None else args.z_max
            grouped_z = {
                name: z[(z >= z_min) & (z <= z_max)]
                for name, z in grouped_z.items()
            }

        non_empty = [z for z in grouped_z.values() if z.size > 0]
        if not non_empty:
            raise ValueError("No labeled z values found after filtering")

        all_labeled_z = np.concatenate(non_empty)
        _, bin_edges = np.histogram(all_labeled_z, bins=args.bins, range=None)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5

        group_counts = {}
        group_smoothed = {}
        group_peaks = {}
        for name, z in grouped_z.items():
            counts, _ = np.histogram(z, bins=bin_edges)
            smoothed = smooth_counts(counts, args.smooth_window)
            group_counts[name] = counts
            group_smoothed[name] = smoothed
            group_peaks[name] = find_peaks(
                bin_centers,
                smoothed,
                args.max_peaks,
                args.min_prominence_ratio,
                args.min_peak_distance,
            )

        print_group_summary(grouped_z, point_counts, bin_files, group_peaks)
        output = plot_group_histogram(bin_edges, group_counts, group_smoothed, group_peaks, args, bin_files)
        print(f"[INFO] wrote plot: {output}")
        return

    z, point_counts = load_z_values(bin_files)
    if args.z_min is not None or args.z_max is not None:
        z_min = -np.inf if args.z_min is None else args.z_min
        z_max = np.inf if args.z_max is None else args.z_max
        z = z[(z >= z_min) & (z <= z_max)]
        if z.size == 0:
            raise ValueError("No z values left after z-range filtering")

    counts, bin_edges = np.histogram(z, bins=args.bins, range=None)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    smoothed = smooth_counts(counts, args.smooth_window)
    peaks = find_peaks(
        bin_centers,
        smoothed,
        args.max_peaks,
        args.min_prominence_ratio,
        args.min_peak_distance,
    )

    print_summary(z, point_counts, bin_files, peaks)
    output = plot_histogram(z, bin_edges, counts, smoothed, peaks, args, bin_files)
    print(f"[INFO] wrote plot: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot z-axis distribution and mark peaks for .bin point clouds.")
    parser.add_argument("inputs", nargs="+",
                        help="One or more .bin files, directories, or glob patterns")
    parser.add_argument("--label-folder", default="",
                        help="Optional folder containing .label files with same stem as .bin")
    parser.add_argument("--label-values", default="curb:3,road:40",
                        help="Classes to plot from labels, e.g. curb:3,road:40")
    parser.add_argument("-o", "--output", default="/home/ant/CurbNet/z_distribution/bin_z_distribution.png",
                        help="Output image path, default /home/ant/CurbNet/z_distribution/bin_z_distribution.png")
    parser.add_argument("--bins", type=int, default=160,
                        help="Histogram bin count")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Odd moving-average window for peak detection; 1 disables smoothing")
    parser.add_argument("--max-peaks", type=int, default=5,
                        help="Maximum number of peaks to mark")
    parser.add_argument("--min-prominence-ratio", type=float, default=0.03,
                        help="Minimum peak prominence as ratio of max count")
    parser.add_argument("--min-peak-distance", type=float, default=0.2,
                        help="Minimum z distance between marked peaks")
    parser.add_argument("--z-min", type=float, default=None,
                        help="Optional minimum z to include")
    parser.add_argument("--z-max", type=float, default=None,
                        help="Optional maximum z to include")
    parser.add_argument("--dpi", type=int, default=140)
    main(parser.parse_args())
