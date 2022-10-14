# Training Smoke Segmentation Model

Training [UNet++](https://arxiv.org/pdf/1807.10165.pdf) using [Smoke5K dataset](https://drive.google.com/file/d/11TM8hsh9R6ZTvLAUzfD6eD051MbOufCi/view?usp=sharing) from [Transmission-BVM](https://github.com/redlessme/Transmission-BVM) to make masks of smoke in wildfire images


Original SMOKE5K dataset
```
SMOKE5K/
    ├── test/
    │   ├─ gt_/
    │   │  └─ (400 masks)
    │   └─ img/
    │      └─ (400 real life smoke)
    └── train/
        ├─ gt_/
        │  ├─ (4000 masks)
        │  └─ (960 masks)
        └─ img/
           ├─ (4000 synthetic smoke)
           └─ (960 real life smoke)

```

[UNet++ : A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/pdf/1807.10165.pdf).

## Installation
1. Create an anaconda environment.
```sh
conda create -n=<env_name> python=3.6 anaconda
conda activate <env_name>
```
2. Install PyTorch.
```sh
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
3. Install pip packages.
```sh
pip install -r requirements.txt
```
## Training on original dataset
Make sure to put the files as the following structure (e.g. the number of classes is 2):
```
inputs
└── <dataset name>
    ├── images
    │   ├── 0a7e06.jpg
    │   ├── 0aab0a.jpg
    │   ├── 0b1761.jpg
    │   ├── ...
    |
    └── masks
        ├── 0
        |   ├── 0a7e06.png
        |   ├── 0aab0a.png
        |   ├── 0b1761.png
        |   ├── ...
        |
        └── 1
            ├── 0a7e06.png
            ├── 0aab0a.png
            ├── 0b1761.png
            ├── ...
```

Real life only data set from SMOKE5K
```
inputs/
└── SMOKE5K/
   ├─ images/
   │    └── (1360 real life smoke)
   └─ masks/
        └── 0/
            └── (1360 masks)
```

1. Train the model.
```
python train.py --dataset <dataset name> --arch NestedUNet --img_ext .jpg --mask_ext .png
```
2. Evaluate.
```
python val.py --name <dataset name>_NestedUNet_woDS
```
## Making masks 
1. Make dataset folder in the following structure

3. Use segmentation.ipynb to make masks folder for each number folder in confirmed/images/
