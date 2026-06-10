import argparse

import numpy as np
import yaml


def load_color_map(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    color_map = {}
    for label, bgr in cfg["color_map"].items():
        color_map[int(label)] = np.asarray(bgr[::-1], dtype=np.uint8)
    return color_map


def labels_to_colors(labels, color_map):
    colors = np.full((labels.shape[0], 3), 120, dtype=np.uint8)
    for label in np.unique(labels):
        colors[labels == label] = color_map.get(int(label), np.asarray([120, 120, 120], dtype=np.uint8))
    return colors


def write_ascii_ply(path, points, colors):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Export SemanticKITTI-style labels as a colored PLY point cloud.")
    parser.add_argument("--bin", required=True, help="Path to .bin point cloud file.")
    parser.add_argument("--label", required=True, help="Path to .label file.")
    parser.add_argument("--out", required=True, help="Output .ply path.")
    parser.add_argument("--yaml", default="config/label_mapping/semantic-kitti.yaml", help="Label mapping yaml.")
    args = parser.parse_args()

    points = np.fromfile(args.bin, dtype=np.float32).reshape((-1, 4))[:, :3]
    labels = np.fromfile(args.label, dtype=np.uint32).reshape((-1,)) & 0xFFFF

    if points.shape[0] != labels.shape[0]:
        raise ValueError(f"Point/label count mismatch: {points.shape[0]} points vs {labels.shape[0]} labels")

    color_map = load_color_map(args.yaml)
    colors = labels_to_colors(labels, color_map)
    write_ascii_ply(args.out, points, colors)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
