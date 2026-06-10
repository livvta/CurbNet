import argparse

import numpy as np
import open3d as o3d
import yaml


def load_color_map(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    color_map = {}
    for label, bgr in cfg["color_map"].items():
        color_map[int(label)] = np.asarray(bgr[::-1], dtype=np.float32) / 255.0
    return color_map


def labels_to_colors(labels, color_map):
    colors = np.full((labels.shape[0], 3), 0.45, dtype=np.float32)
    for label in np.unique(labels):
        colors[labels == label] = color_map.get(int(label), np.asarray([0.45, 0.45, 0.45], dtype=np.float32))
    return colors


def main():
    parser = argparse.ArgumentParser(description="Visualize SemanticKITTI-style point labels with Open3D.")
    parser.add_argument("--bin", required=True, help="Path to .bin point cloud file.")
    parser.add_argument("--label", required=True, help="Path to .label file.")
    parser.add_argument("--yaml", default="config/label_mapping/semantic-kitti.yaml", help="Label mapping yaml.")
    parser.add_argument("--point-size", type=float, default=1.0, help="Open3D point size.")
    args = parser.parse_args()

    points = np.fromfile(args.bin, dtype=np.float32).reshape((-1, 4))[:, :3]
    labels = np.fromfile(args.label, dtype=np.uint32).reshape((-1,)) & 0xFFFF

    if points.shape[0] != labels.shape[0]:
        raise ValueError(f"Point/label count mismatch: {points.shape[0]} points vs {labels.shape[0]} labels")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(labels_to_colors(labels, load_color_map(args.yaml)))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"{args.bin} + {args.label}", width=1280, height=720)
    vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = args.point_size
    render_opt.background_color = np.asarray([0.02, 0.02, 0.02])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
