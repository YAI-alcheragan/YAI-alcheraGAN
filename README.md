# YAI-Alchera Project : Wildfire Generator


> ***Repository*** : https://github.com/YAI-alcheragan/YAI-alcheraGAN

## Credits

**[YAI](https://www.instagram.com/yonsei.ai) 10th 2022 Summer Project - AlcehraGAN Team**

* Team Leader : **[Kihyun Kim](https://github.com/)** - Yonsei Univ. College of .

* Team Member : **[Donggeon Bae](https://github.com/AttiBae)** - Yonsei. Univ. Dept. of Electrical and Electronic Engineering.

* Team Member : **[Subin Kang](https://github.com/)** - Yonsei Univ. Dept. .

* Team Member : **[Jungbin Joe](https://github.com/)** - Yonsei Univ. Dept. .

* Team Member : **[Sangheon Lee](https://github.com/)** - Yonsei Univ. Dept. .

* Team Member : **[Minjae Lee](https://github.com/)** - Yonsei Univ. Dept. .

---

## Dataset

> ***Wildfire Dataset from Alchera : https://alchera.ai/company/about-alchera***

### Data Description

This dataset, also known as PanNuke, contains semi automatically generated nuclei instance segmentation and classification images with exhaustive nuclei labels across 19 different tissue types. The dataset consists of 481 visual fields, of which 312 are randomly sampled from more than 20K whole slide images at different magnifications, from multiple data sources.

In total the dataset contains 205,343 labeled nuclei, each with an instance segmentation mask. Models trained on PanNuke can aid in whole slide image tissue type segmentation, and generalise to new tissues.

19 Tissue types (Breast, Colon, Bile-duct, Esophagus, Uterus, Lung, Cervix, Head&Neck, Skin, Adrenal Gland, Kidney, Stomach, Prostate, Testis, Liver, Thyroid, Pancreas, Ovary, Bladder). Note, that it also unifies existing datasets within it, we have carefully labelled these under a single nuclei categories schema that is common to all 19 tissues.

This particular directory includes training patches of size 256x256 and their masks, this is one of the folds. In total there are more than 7 thousand training patches within all the three folds.

The files within each fold directories are:

* `images.npy` - image patches of 256x256

* `masks.npy` an array of 6 channel instance-wise masks (0: Neoplastic cells, 1: Inflammatory, 2: Connective/Soft tissue cells, 3: Dead Cells, 4: Epithelial, 6: Background)

* `types.npy`  tissue type that a given path was extracted from.

### Data Preview

![Preview](./assets/asset1.png)

### Origin: Kaggle

* [**#1**](https://www.kaggle.com/andrewmvd/cancer-inst-segmentation-and-classification): 12.53GiB

* [**#2**](https://www.kaggle.com/andrewmvd/cancer-instance-segmentation-and-classification-2): 11.91GiB

* [**#3**](https://www.kaggle.com/andrewmvd/cancer-instance-segmentation-and-classification-3): 12.84GiB

---

## Model Architecture

**GP-GAN**: Backbone

* **Paper**: [NEED Arxiv CITATION](NEED LINK)

* **Implementation**: [Pytorch Vision](NEED LINK)

**Wildfire Segmentation**: Generate Mask

* **Paper**: [NEED Arxiv CITATION](NEED LINK)

* **Implementation**: [Pytorch Vision](NEED LINK)

**Color Match**: Determine Mask Postion

* **Paper**: [NEED Arxiv CITATION](NEED LINK)

* **Implementation**: [Pytorch Vision](NEED LINK)

## Metrics


- Cost function - Hybrid Loss

  $$
  \text{Loss} = 2\times \text{BCE } + 2 \times \text{Dice } + \text{IoU}
  $$

  1. **Binary Cross Entropy**

  $$
  \text{BCE} = - \sum _{i=1} ^{\text{output size}} y_i \cdot \log {\hat{y}_i}
  $$

  
- Optimizing - **Stochastic Gradient Descent with Adam(IF IT ANALISED)**

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

* Github: [https://github.com/YAI-alcheragan/YAI-alcheraGAN]()

All non-necessary codes are modularized as package. Watch all codes in github repository.
