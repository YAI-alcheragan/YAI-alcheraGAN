
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
    |   ├── 0a7e06.jpg
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

1. Train the model.
```
python train.py --dataset <dataset name> --arch NestedUNet --img_ext .jpg --mask_ext .png
```
2. Evaluate.
```
python val.py --name <dataset name>_NestedUNet_woDS
