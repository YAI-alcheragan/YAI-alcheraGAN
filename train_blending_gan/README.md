
### Training Setting
- pytorch 1.9.0
- numpy 1.20.3
- torchvision 0.10.0
- tqdm 4.62.2
- pillow 8.3.1
- opencv-python 4.5.3.56


### Data Preview
To inference, make datasets folder consisting of images, labels, masks per each file #
Make sure to put the files as the following structure 

### To train, 

**split datasets to train and test**:
```
pyrhon train_test_split.py

After run this code your datasets must follow the structure below

└── datasets/
    ├── train/
    │   ├── bg/
    │   │   └── ...
    │   └── smoke/
    │       └── ...
    ├── test
    │   ├── bg/
    │   │   └── ...
    │   └── smoke/
    │       └── ...
    
```

**GP-GAN**:
```
python train.py --data_root "{root path of datasets}" --bg_dir "bg" --obj_dir "smoke" --size 64 --pretrained "{path of pretrained model}" --batch_size 64 
```

