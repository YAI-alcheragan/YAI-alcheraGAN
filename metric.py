from metric_dataset import MetricDataset
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision
from torchvision import models, transforms

from prdc import compute_prdc
import numpy as np
import random

# random seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def cal_metric (dataroot_true, dataroot_false, using_img_num):
  
  true_dataset, false_dataset = MetricDataset(dataroot_true), MetricDataset(dataroot_false)
  true_loader = DataLoader(true_dataset, batch_size=using_img_num, shuffle=True, drop_last=True)
  false_loader = DataLoader(true_dataset, batch_size=using_img_num, shuffle=True, drop_last=True)

  true_images, false_images = next(iter(true_loader)).to(torch.uint8), next(iter(false_loader)).to(torch.uint8)
  
  # cal fid 
  fid = FrechetInceptionDistance(feature=2048) #64, 192, 768, 2048 가능
  fid.update(true_images, real=True)
  fid.update(false_images, real=False)
  fid_score = fid.compute()
  
  # cal density & coverage
  use_pretrained= True
  vgg_net = models.vgg16(pretrained=use_pretrained)
  nearest_k = 5

  with torch.no_grad():
    true_vgg_feature = vgg_net(true_images.float())
    false_vgg_feature = vgg_net(false_images.float())
    
  dc_score = compute_prdc(real_features=true_vgg_feature,
                        fake_features=false_vgg_feature,
                        nearest_k=2)
  
  return fid_score, dc_score['density'], dc_score['coverage']

