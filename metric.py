from metric_dataset import MetricDataset
import torch
from torch.utils.data import DataLoader

from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision
from torchvision import models, transforms

from prdc import compute_prdc
import numpy as np
import random

import argparse

'''random seed'''
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def cal_metric (dataroot_true, dataroot_false, using_img_num):
  """
  FID: Pre-trained inception model을 이용하여 Real world samples와 generated samples의 embedding 분포를 구하여 Frechet distance를 계산하는 지표
  Density & Coverage: Pre-trained 혹은 randomly initialized VGG16을 통해 embedding을 구하여 fidelity 와 diversity를 구분하여 평가하는 지표 
  """  
  '''dataloader: true_loader(Alchera images), false_loader(generated images)'''
  true_dataset, false_dataset = MetricDataset(dataroot_true), MetricDataset(dataroot_false)
  true_loader = DataLoader(true_dataset, batch_size=using_img_num, shuffle=True, drop_last=True)
  false_loader = DataLoader(false_dataset, batch_size=using_img_num, shuffle=True, drop_last=True)

  true_images, false_images = next(iter(true_loader)).to(torch.uint8), next(iter(false_loader)).to(torch.uint8)
  
  '''calculate FID'''
  fid = FrechetInceptionDistance(feature=2048) #64, 192, 768, 2048 available
  fid.update(true_images, real=True)
  fid.update(false_images, real=False)
  fid_score = fid.compute()
  
  '''calculate Density & Converage'''
  use_pretrained= True
  vgg_net = models.vgg16(pretrained=use_pretrained)

  with torch.no_grad():
    true_vgg_feature = vgg_net(true_images.float())
    false_vgg_feature = vgg_net(false_images.float())
    
  dc_score = compute_prdc(real_features=true_vgg_feature,
                        fake_features=false_vgg_feature,
                        nearest_k=2)
  
  return fid_score, dc_score['density'], dc_score['coverage']

if __name__=="__main__":
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--true_data_root', 
                      help='data root for true images')
  parser.add_argument('--false_data_root',
                      help='data root for false images')
  parser.add_argument('--calculate_num', type=int,
                        help='number of images to use to calculate metric')


  args = parser.parse_args()
