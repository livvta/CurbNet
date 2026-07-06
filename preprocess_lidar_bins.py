"""
python preprocess_lidar_bins.py \
  --src /path/to/mrs/velodyne \
  --dst /path/to/mrs_preprocessed/velodyne
"""
#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def preprocess_points(points, max_radius=50.0, min_z=-4.0, max_z=2.0):
    if points.shape[0] == 0:
        return points

    points = points.copy()
    if points[:, 3].max() > 1.0:
        points[:, 3] /= 255.0

    radius = np.linalg.norm(points[:, :2], axis=1)
    mask = (
        (radius >= 0.0) & (radius <= max_radius) &
        (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
    )
    return points[mask]


def load_bin(path):
    points = np.fromfile(path, dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(f"{path} is not a float32 Nx4 bin file")
    return points.reshape(-1, 4)


def output_path_for(src_file, src_root, dst_root):
    if src_root.is_file():
        if dst_root.suffix == ".bin":
            return dst_root
        return dst_root / src_file.name
    return dst_root / src_file.relative_to(src_root)


def iter_bin_files(src, recursive=False):
    if src.is_file():
        if src.suffix != ".bin":
            raise ValueError(f"{src} is not a .bin file")
        yield src
        return

    pattern = "**/*.bin" if recursive else "*.bin"
    yield from sorted(src.glob(pattern))


def preprocess_file(src_file, dst_file, args):
    points = load_bin(src_file)
    processed = preprocess_points(
        points,
        max_radius=args.max_radius,
        min_z=args.min_z,
        max_z=args.max_z,
    )

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    processed.astype(np.float32).tofile(dst_file)

    kept = processed.shape[0]
    total = points.shape[0]
    pct = kept / total * 100 if total else 0.0
    print(f"{src_file} -> {dst_file} | kept {kept:,}/{total:,} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Normalize intensity and crop float32 Nx4 LiDAR .bin files."
    )
    parser.add_argument("--src", required=True, help="Input .bin file or folder.")
    parser.add_argument("--dst", required=True, help="Output .bin file or folder.")
    parser.add_argument("--recursive", action="store_true", help="Process subfolders too.")
    parser.add_argument("--max-radius", type=float, default=50.0)
    parser.add_argument("--min-z", type=float, default=-4.0)
    parser.add_argument("--max-z", type=float, default=2.0)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise FileNotFoundError(src)

    bin_files = list(iter_bin_files(src, recursive=args.recursive))
    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in {src}")

    for src_file in bin_files:
        dst_file = output_path_for(src_file, src, dst)
        preprocess_file(src_file, dst_file, args)


if __name__ == "__main__":
    main()
