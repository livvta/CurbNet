
## CurbNet: Curb Detection Framework Based on LiDAR Point Cloud Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2403.16794-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2403.16794)
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-red.svg?style=flat)](https://github.com/guoyangzhao/CurbNet/pulls)


---

## 📌 Overview

Curb detection is a critical task in autonomous driving and robotic perception, enabling accurate understanding of road boundaries and drivable areas.

However, it remains challenging due to:
- (a) Difficulty in extracting **height-aware geometric features**
- (b) **Non-uniform density distribution** of LiDAR point clouds
- (c) Severe **class imbalance** between curb and non-curb points

To address these challenges, we propose **CurbNet**, a LiDAR-based curb detection framework with:

- A newly constructed **3D-Curb dataset**
- A **Multi-Scale Channel Attention (MSCA)** module for feature fusion and height modeling
- A **loss group strategy** to handle class imbalance
- A **post-processing refinement module** for improved prediction quality



## 🧠 Framework

<img src="https://github.com/guoyangzhao/CurbNet/blob/main/images/framework527.png" width="65%">


## 📊 Detection Results

### SemanticKITTI Benchmark

<img src="https://github.com/guoyangzhao/CurbNet/blob/main/images/3Dcurb-no-occ2.png" width="55%">



## 🗂️ Dataset: 3D-Curb

We construct the **3D-Curb dataset** based on the large-scale **SemanticKITTI dataset**, with the following features:

- Add a **new curb category (label = 3)** with full 3D annotations
- Preserve original **28 semantic classes**
- Captured using **64-line LiDAR**
- Designed for **urban autonomous driving scenarios**

<img src="https://github.com/guoyangzhao/CurbNet/blob/main/images/Dataset_construct527.png" width="55%">



### 📥 Download

- **Baidu Netdisk**: [Download Link](https://pan.baidu.com/s/1YKtiCdgugzTxHD6dTpXNHQ) (Code: 1234)  
- **Google Drive**: [Download Link](https://drive.google.com/file/d/18rK1TE96SfWMi2GjK0RO1U4UPxumA1-E/view?usp=sharing)



### 📦 Data Format

The dataset follows the **SemanticKITTI format**:

```

/kitti/dataset/
└── sequences/
├── 00/
│   ├── labels/
│   │     ├ 000000.label
│   │     └ 000001.label
│   └── velodyne/
│         ├ 000000.bin
│         └ 000001.bin
├── 01/
├── 02/
...
└── 10/

```

- Point clouds: `.bin`
- Labels: `.label`



### ⚠️ Important Note

Since **curb is an additional category**, we assign:

```

label: 3 → "curb"

```

To visualize correctly using SemanticKITTI API, please modify:

```

config/semantic-kitti.yaml

```

Add:

```

3: "curb"

```



### 🔧 Visualization Tool

We recommend using the official SemanticKITTI API:

👉 https://github.com/PRBonn/semantic-kitti-api



## 📁 Additional Dataset: NRS

The original **NRS dataset** contains only 2D annotations.

We:
- Project 2D labels to **3D point clouds**
- Convert format to **SemanticKITTI style**
- Release processed version for public use

### 📥 Download

- **Baidu Netdisk**: [Download Link](https://pan.baidu.com/s/1U6b5c6TxfruNT572_vwLYA) (Code: 1234)  
- **Google Drive**: [Download Link](https://drive.google.com/file/d/1kAj1xEHnppwrg2zLp42rCsLL8439glhy/view?usp=sharing)


---


## ⚙️ Installation

### Requirements

- Python ≥ 3.7
- PyTorch ≥ 1.2
- yaml
- Cython
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)
- [spconv](https://github.com/traveller59/spconv) (tested with `spconv==1.2.1`, `cuda==10.2`)
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) *(optional)*



## 🚀 Training

1. Modify config file:

```

config/semantickitti-curb_0.2.yaml

````

2. Start training:

```bash
sh train_0.2.sh
````



## 🎯 Inference Demo

### Run demo on a folder of LiDAR scans:

```bash
python demo_folder_focal.py \
    --demo-folder YOUR_FOLDER \
    --save-folder YOUR_SAVE_FOLDER
```

### With labels (optional):

```bash
python demo_folder_focal.py \
    --demo-folder YOUR_FOLDER \
    --save-folder YOUR_SAVE_FOLDER \
    --demo-label-folder YOUR_LABEL_FOLDER
```

---

## ⭐ Citation

If you find this project useful, please consider giving a ⭐ and citing our work:

```bibtex
@article{zhao2025curbnet,
  title={CurbNet: Curb detection framework based on LiDAR point cloud segmentation},
  author={Zhao, Guoyang and Ma, Fulong and Qi, Weiqing and Liu, Yuxuan and Liu, Ming and Ma, Jun},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE}
}
```



## 📬 Contact

If you have any questions or suggestions, feel free to open an issue or contact the authors.



## 🌟 Acknowledgements

This project is built upon:

* SemanticKITTI dataset
* Open-source LiDAR perception frameworks [Cylinder3D](https://github.com/xinge008/Cylinder3D)

We thank the community for their contributions.
