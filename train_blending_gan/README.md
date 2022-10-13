
## Training Setting
- pytorch 1.9.0
- numpy 1.20.3
- torchvision 0.10.0
- tqdm 4.62.2
- pillow 8.3.1
- opencv-python 4.5.3.56

## To train

**Split datasets to train and test**:
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

**Training GP-GAN**:
```
python train.py --data_root "{root path of datasets}" --bg_dir "bg" --obj_dir "smoke" --size 64 --pretrained "{path of pretrained model}" --batch_size 64 
```

## Problems
### Fine-tuning using pretrained model
We uses pretrained model from https://github.com/wuhuikai/GP-GAN and the source codes are developed using "chainer" framework not pytorch.
Since we do not know any information of chainer, we ported chainer code to pytorch, and in the progression of loading pretrained model, which is also built on chainer framework, we found some weight information loss. That is why result of pretrained model or fine-tuned model shows poor quality.

### Traininig GP-GAN from scratch
We also tried to train model from scratch, not using pretrained model. But the results are way worse than pretrained and fine-tuned model.
We suppose that not enough data might be the reason for failure.
