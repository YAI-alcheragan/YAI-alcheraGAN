# YAI-Alchera Project : Wildfire Image Generator

> ***Repository*** : https://github.com/YAI-alcheragan/YAI-alcheraGAN

## Credits

**[YAI](https://www.instagram.com/yonsei.ai) 10th 2022 Summer Project - AlcehraGAN Team**

* Team Leader : **[Gihyun Kim](https://github.com/gihyunkim)** - Yonsei Univ. Dept of Artificial Intelligence.

* Team Member : **[Donggeon Bae](https://github.com/AttiBae)** - Yonsei. Univ. Dept. of Electrical and Electronic Engineering.

* Team Member : **[Jungbin Cho](https://github.com/whwjdqls)** - Yonsei. Univ. Dept. of Electrical and Electronic Engineering.

* Team Member : **[Minjae Lee](https://github.com/98minjae)** - Yonsei Univ. Dept. of Creative Technology Management.

* Team Member : **[Sangheon Lee](https://github.com/lsh-159)** - Yonsei. Univ. Dept. of Electrical and Electronic Engineering.

* Team Member : **[Subin Kang](https://github.com/suuuuuuuubin)** - Yonsei Univ. Dept. Computer Science.

---
## Installation

1. Create an anaconda environment.
```sh
conda create -n <env_name> 
conda activate <env_name>
```
2. Install PyTorch
```sh
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```
4. git clone
```sh
git clone https://github.com/YAI-alcheragan/YAI-alcheraGAN.git
```
3. Install pip packages.
```sh
pip install -r requirements.txt
```


## Dataset

> ***Wildfire Dataset Taken from Alchera Wildfire Detection : https://alchera.ai/***

### Data Description

This dataset contains wildfire images that are given as detected sets from Alchera, AI technology development company based on face & anomalous situation detection (e.g. wildfire).

In total the dataset contains 20k wild forest images of size 640x480: 10k in confirmed(detected as wildfire situation) and the other 10k in skipped(detected as nothing-happened) as jpg images, each with a bbox position for detective area that is classified as a situation as JSON file

We generated segmentation mask dataset folder for the confirmed dataset mentioned above, each has same label(image name) for a target of segmentation. This particular directory includes segmented binary masks of size 640x480, following original target image's. A actual area of smoke has size of 128x128 which it's center position is derived from a given image's bbox

Additional dataset called Smoke5K, which used for segmentation model train([UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165)), has 5,400 images that are mixture of real(1,400) and synthetic(4,000) smoke images. For preventing overfitting of the segmentation and maintaining naturality of the segmented smoke from general images, we tried to train the model with an unaffective smoke set so we chose this dataset.

The files within each fold directories except the additional dataset are:

* `confirmed/#/images/` - directory contains 10k of 640x480 jpg images

* `confirmed/#/labels/` - directory contains 10k of json labels for the directory mentioned above ([left, up, right, bottom]

* `confirmed/#/masks/` - directory contains 10k of segmented masks for the directory mentioned above

* `skipped/` - directory contains 10k of 640x480 normal wild forest images

### Data Preview
To inference, make datasets folder consisting of images, labels, masks per each file #
Make sure to put the files as the following structure 
```
└── YAI-alcheraGAN/
    ├── datasets/
    │   ├── confirmed/
    │   │   ├── 1/
    │   │   │   ├── images/
    │   │   │   │   └── (labelfilenames)
    │   │   │   ├── labels/
    │   │   │   │   └── (labelfilenames)
    │   │   │   └── masks/
    │   │   │       └── (labelfilenames)
    │   │   └── ...
    │   └── skipped/
    │       ├── 1/
    │       │   └── (labelfilenames)
    │       └── ...
    ├── train_segmentation
    ├── gp_gan.py
    └── ...
```
masks folder made by [train_segementation](https://github.com/YAI-alcheragan/YAI-alcheraGAN/tree/main/train_segmentation)
### Origin

* [**Alchera**](https://alchera.ai/)

* [**Smoke5K**](https://ojs.aaai.org/index.php/AAAI/article/view/20207) - [**Github**](https://github.com/redlessme/Transmission-BVM)

---

## Model Architecture
![Pipeline](https://github.com/YAI-alcheragan/YAI-alcheraGAN/blob/main/img/Snipaste_2022-10-14_12-24-52.png)

**GP-GAN**: Backbone

* **Paper**: [ArXiv 1703.07195](https://arxiv.org/abs/1703.07195)

* **Implementation**: [Pytorch Vision](https://github.com/wuhuikai/GP-GAN)

**Wildfire Segmentation**: Generate Mask

* **Paper**: [ArXiv 1807.10165](https://arxiv.org/abs/1807.10165)

* **Implementation**: [Pytorch Vision](https://github.com/4uiiurz1/pytorch-nested-unet)

**Color Match**: Determine Mask Postion Depending on Color Distribution

* **Implementation**: [OpenCV](https://github.com/kb22/Color-Identification-using-Machine-Learning)

## Metrics

  1. **FID**

  $$
  \text{FID} = ||\mu_x-\mu_y ||^2 + \text{Tr}(\Sigma_\text{x} + \Sigma_\text{y} - 2\sqrt{\Sigma_\text{x}\Sigma_\text{y}})
  $$

  2. **KID**
  
  $$
  \text{KID} = E_{x, x^{\prime}p}[K(x,x^{\prime})]+E_{x,x^{\prime}q}[K(x,x^{\prime})]-2E_{xp,x^{\prime}p}[K(x,x^{\prime})]
  $$
  
  
  [**Density and Convergence**](https://github.com/clovaai/generative-evaluation-prdc):
  
  3. **Density**
  
  $$
  \text{Density} := \frac{1}{kM}\sum_{j=1}^{M}\sum_{i=1}^{N}1_{Y_j\in B(X_i,\text{NND}_k(X_i))}
  $$
  
  4. **Converage**
  
  $$
  \text{Coverage} := \frac{1}{N}\sum_{i=1}^{N}1_{\exists\text{ }j\text{ s.t. } Y_j\in B(X_i,\text{NND}_k(X_i))}
  $$

---

## Results

### Source Example

![Source Example](/img/img1.PNG)

### Segmentation Example

![Segmentation Example](/img/img2.PNG)

### Target Example

![Target Example](/img/img3.PNG)

### Final Output

![Final Output](/img/img4.PNG)

## Experiment

### Experiment description

![Target Example](/img/Experiment_Description.PNG)

### Experiment examples

![Source Example](/img/experiment1.PNG)

![Source Example](/img/experiment2.PNG)

![Source Example](/img/experiment3.PNG)

![Source Example](/img/experiment4.PNG)

![Source Example](/img/experiment5.PNG)

---

## Pretrained models



**GP-GAN**: Blending GAN G(x)

* **blending_gan.npz**: [download](https://drive.google.com/uc?export=download&id=11L-n9cERvQJXkOLYGaOS-jyAt6gH6mHq)

* **finetuned.pt**: [download](https://drive.google.com/uc?export=download&id=10dDiysI4JKo7U47Gvor1m2d1-iRXbUwp)

**Wildfire Segmentation**: Unet++

* **pretrained.pth**: [download](https://drive.google.com/uc?export=download&id=1kiPLkQmu51w2zc_LC3qWigTuONK1SkZf)


---

## Full Source Code

* Github: [https://github.com/YAI-alcheragan/YAI-alcheraGAN]

All non-necessary codes are modularized as package. Watch all codes in github repository.
=======

### To train, 

---

**Wildfire Segmentation**:  [train_segmentation](https://github.com/YAI-alcheragan/YAI-alcheraGAN/tree/main/train_segmentation)

**GP-GAN**: [train_blending_gan](https://github.com/YAI-alcheragan/YAI-alcheraGAN/tree/main/train_blending_gan)



### To inference, 

---
1. Create **Segmentation Mask** Images

**Wildfire Segmentation**: [inference_segmentation](https://github.com/YAI-alcheragan/YAI-alcheraGAN/tree/main/train_segmentation)

2. Run Model
**GP-GAN**:
```
python inference_gpgan.py --root ./datasets --result_folder ./results --experiment_folder ./experiments --g_path ./blending_gan.npz 
```


### To Compute Metrics, 

---
1. Set True images and False(Generated) images Folder
```
└── true_data_root/
    ├── true_image1.jpg
    ├── true_image2.jpg
    └── ...
└── false_data_root/
    ├── true_image1.jpg
    ├── true_image2.jpg
    └── ...
```
2. Compute Metrics
```
!python metric.py --true_data_root "real image root" --false_data_root "generated image root" --calculate_num "number of images to use to cal metrics" 
```
