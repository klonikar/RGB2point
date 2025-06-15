<p align="center">
    <img src="https://www.jaejoonglee.com/images/rgb2point.png" alt="Overview">
</p>

**RGB2Point** is officially accepted to WACV 2025. It takes a single unposed RGB image to generate 3D Point Cloud. Check more details from the [paper](https://arxiv.org/pdf/2407.14979).
## Codes
**RGB2Point** is tested on Ubuntu 22 and Windows 11. `Python 3.9+` and `Pytorch 2.0+` is required.

## Dependencies
Assuming `Pytorch 2.0+` with `CUDA` is installed, run:
```
pip install timm
pip install accelerate
pip install wandb
pip install open3d
pip install scikit-learn
pip install open3d
```

## Training
```
python train.py
```

## Training Data
Please download 1)  [Point cloud data zip file](https://drive.google.com/file/d/1R7TXnBvVir8OCXPE5f2kck6Enl0gdMUQ/view?usp=sharing), 2) [Rendered Images](https://drive.google.com/file/d/1t_rlV1BwitvICap_2ubd5oqL_6Yq-Drn/view?usp=sharing), and 3) [Train/test filenames](https://drive.google.com/drive/folders/1jBPd1YBJwzgVpolT-yA0g8XxYJmb2_s-?usp=sharing).

Next, modify the downloaded 1), 2), 3) file paths to [L#36](https://github.com/JaeLee18/RGB2point/blob/7b29188ea8b4c92fcc5f48bd0066e901881ce1f7/utils.py#L36), [L#38](https://github.com/JaeLee18/RGB2point/blob/7b29188ea8b4c92fcc5f48bd0066e901881ce1f7/utils.py#L38), [L#14](https://github.com/JaeLee18/RGB2point/blob/7b29188ea8b4c92fcc5f48bd0066e901881ce1f7/utils.py#L14) and [L#16](https://github.com/JaeLee18/RGB2point/blob/7b29188ea8b4c92fcc5f48bd0066e901881ce1f7/utils.py#L16).

## Pretrained Model
Download the model trained on Chair, Airplane and Car from ShapeNet.
```
https://drive.google.com/file/d/1Z5luy_833YV6NGiKjGhfsfEUyaQkgua1/view?usp=sharing
```

## Inference
```
python inference.py
```
Change `image_path` and `save_path` in `inference.py` accrodingly.


## update
We have added support for the BlendedMVS Dataset with a custom DataLoader located in the notebooks folder. This integration enhances multi-view stereo capability for 3D reconstruction.

## Reference
If you find this paper and code useful in your research, please consider citing:
```bibtex
@InProceedings{Lee_2025_WACV,
    author    = {Lee, Jae Joong and Benes, Bedrich},
    title     = {RGB2Point: 3D Point Cloud Generation from Single RGB Images},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {2952-2962}
}
```
