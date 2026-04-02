
# CurbNet: Curb Detection Framework Based on LiDAR Point Cloud Segmentation


## Installation

### Requirements
- PyTorch >= 1.2 
- yaml
- Cython
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) (optional for nuScenes)
- [spconv](https://github.com/traveller59/spconv) (tested with spconv==1.2.1 and cuda==10.2)

## Data Preparation

### 3D-Curb Datasets
The data is organized in the following format:
```
/kitti/dataset/
          └── sequences/
                  ├── 00/
                  │   ├── labels/
                  │   │     ├ 000000.label
                  │   │     └ 000001.label
                  │   └── velodyne/
                  │         ├ 000000.bin
                  │         └ 000001.bin
                  ├── 01/
                  ├── 02/
                  .
                  └── 10/
```



## Training
1. modify the config/semantickitti-curb_0.2.yaml with your custom settings. We provide a sample yaml for 3D-Curb Datasets
2. train the network by running "sh train_0.2.sh"


## Semantic segmentation demo for a folder of lidar scans
```
python demo_folder_focal.py --demo-folder YOUR_FOLDER --save-folder YOUR_SAVE_FOLDER
```
If you want to validate with your own datasets, you need to provide labels.
--demo-label-folder is optional
```
python demo_folder_focal.py --demo-folder YOUR_FOLDER --save-folder YOUR_SAVE_FOLDER --demo-label-folder YOUR_LABEL_FOLDER
```

## Citations:
If you find CurbNet or 3D-Curb Dataset useful in your research or applications, please consider giving us a star 🌟 and citing it.

```bibtex
@article{zhao2024curbnet,
  title={CurbNet: Curb Detection Framework Based on LiDAR Point Cloud Segmentation},
  author={Zhao, Guoyang and Ma, Fulong and Liu, Yuxuan and Qi, Weiqing and Liu, Ming and Ma, Jun},
  journal={arXiv preprint arXiv:2403.16794},
  year={2024}
}
```
