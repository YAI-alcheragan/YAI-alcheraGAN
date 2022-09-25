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

## Dataset

> ***Wildfire Dataset Taken from Alchera Wildfire Detection : https://alchera.ai/***

### Data Description

This dataset contains wildfire images that are given as detected sets from Alchera, AI technology development company based on face & anomalous situation detection (e.g. wildfire).

In total the dataset contains 10k wild forest images of size 640x480: 5k in confirmed(detected as wildfire situation) and the other 5k in skipped(detected as nothing-happened) as jpg images, each with a bbox position for detective area that is classified as a situation as JSON file

We generated segmentation mask dataset folder for the confirmed dataset mentioned above, each has same label(image name) for a target of segmentation. This particular directory includes segmented binary masks of size 640x480, following original target image's. A actual area of smoke has size of 128x128 which it's center position is derived from a given image's bbox

Additional dataset called Smoke5K, which used for segmentation model train, has 5,360 images that are mixture of real and synthetic smoke images and the other 7K non-smoke images. For preventing overfitting of the segmentation and maintaining naturality of the segmented smoke from general images, we tried to train the model with an unaffective smoke set so we chose this dataset.

The files within each fold directories are:

* `confirmed/#/images/` - directory contains 5k of 640x480 jpg images

* `confirmed/#/labels/` - directory contains 5k of json labels for the directory mentioned above ([left, up, right, bottom]

* `confirmed/#/masks/` - directory contains 5k of segmented masks for the directory mentioned above

* `skipped/` - directory contains 5k of 640x480 normal wild forest images

### Data Preview

![Preview](NEED LINK)

### Origin: Alchera

* [**!**](https://alchera.ai/)

---

## Model Architecture

**GP-GAN**: Backbone

* **Paper**: [ArXiv 1703.07195](https://arxiv.org/abs/1703.07195)

* **Implementation**: [Pytorch Vision](https://github.com/wuhuikai/GP-GAN)

**Wildfire Segmentation**: Generate Mask

* **Paper**: [NEED Arxiv CITATION](NEED LINK)

* **Implementation**: [Pytorch Vision](NEED LINK)

**Color Match**: Determine Mask Postion

* **Paper**: [NEED Arxiv CITATION](NEED LINK)

* **Implementation**: [Pytorch Vision](NEED LINK)

## Metrics

  1. **FID**

  $$
  \text{FID} = ||\mu_x-\mu_y ||^2 + \text{Tr}(\Sigma_\text{x} + \Sigma_\text{y} - 2\sqrt{\Sigma_\text{x}\Sigma_\text{y}})
  $$

  2. **KID**
  
  $$
  \text{KID} = E_{x, x^{\prime}p}[K(x,x^{\prime})]+E_{x,x^{\prime}q}[K(x,x^{\prime})]-2E_{xp,x^{\prime}p}[K(x,x^{\prime})]
  $$
  
  3. **Density**
  
  $$
  <a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;\text{density}:=\frac{1}{kM}\sum_{j=1}^{M}\sum_{i=1}^{N}1_{Y_j\in&space;B(X_i,\text{NND}_k(X_i))}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\text{Density}:=\frac{1}{kM}\sum_{j=1}^{M}\sum_{i=1}^{N}1_{Y_j\in&space;B(X_i,\text{NND}_k(X_i))}" title="\text{density}:=\frac{1}{kM}\sum_{j=1}^{M}\sum_{i=1}^{N}1_{Y_j\in B(X_i,\text{NND}_k(X_i))}" /></a>
  $$
  
  4. **Converage**
  
  $$
  <a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;\text{Coverage}:=\frac{1}{N}\sum_{i=1}^{N}1_{\exists\text{&space;}j\text{&space;s.t.&space;}&space;Y_j\in&space;B(X_i,\text{NND}_k(X_i))}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\text{coverage}:=\frac{1}{N}\sum_{i=1}^{N}1_{\exists\text{&space;}j\text{&space;s.t.&space;}&space;Y_j\in&space;B(X_i,\text{NND}_k(X_i))}" title="\text{coverage}:=\frac{1}{N}\sum_{i=1}^{N}1_{\exists\text{ }j\text{ s.t. } Y_j\in B(X_i,\text{NND}_k(X_i))}" /></a>
  $$

---

## Results

### Target Example

![Target Example](NEED GITHUB DATA LINK)

### Source Example

![Source Example](NEED GITHUB DATA LINK)

### Segmentation Example

![Segmentation Example](NEED GITHUB DATA LINK)

### Final Output

![Final Output](NEED GITHUB DATA LINK)

---

## Full Source Code

* Github: [https://github.com/YAI-alcheragan/YAI-alcheraGAN]

All non-necessary codes are modularized as package. Watch all codes in github repository.
