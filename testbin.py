import numpy as np, os
# p = "/home/ant/ros2_humble/dataset/industrial_bin/0000000003.bin"
p = "/home/ant/CurbNet/data/3D-Curb/03/velodyne/000093.bin"
# p = "/home/ant/CurbNet/data/NRS/transfer_velodyne/20220705102852_Sunny_City_Day_0001.bin"
size = os.path.getsize(p)
print("file:", p)
print("bytes:", size)
print("float32 count:", size // 4)
print("can reshape Nx4:", size % 16 == 0)
pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
print("shape:", pts.shape)
print("xyz min:", pts[:, :3].min(axis=0))
print("xyz max:", pts[:, :3].max(axis=0))
print("intensity min/max:", pts[:, 3].min(), pts[:, 3].max())
print("first rows:", pts[:5])