import os

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2

class BlendingDataset(Dataset):
  def __init__(self, root, bg_folder, obj_folder, obj_ann, mode, load_size):
    
    self.root = root
    self.bg_folder = os.path.join(root, mode, bg_folder)
    self.obj_folder = os.path.join(root, mode, obj_folder)
    self.mode = mode
    self.load_size = load_size

    self.bg_imgs = os.listdir(self.bg_folder)
    obj_ann_df =  pd.read_csv(os.path.join(self.obj_folder, obj_ann)) #dataset 구조: root/train/smoke/img1.jpg // root/train/smoke/_annotations.csv
    self.obj_anns = obj_ann_df.values.tolist()
    self.obj_anns_len = len(self.obj_anns)
    
    self.transform = self.get_transform()
    
  def __getitem__(self, idx):
    bg = cv2.imread(os.path.join(self.bg_folder,self.bg_imgs[idx]))
    bg_h, bg_w, _ = bg.shape
    random_idx = np.random.randint(len(self.obj_anns))
    # roboflow (Tensorflow Object Detection CSV format)
    obj_path, _, _, _, x1, y1, x2, y2 = self.obj_anns[random_idx] #get random obj img info
    obj = cv2.imread(os.path.join(self.obj_folder,obj_path))
    obj_h, obj_w, _ = obj.shape

    ratio = min(bg_w/obj_w, bg_h/obj_h)   #bg와 obj의 크기 차이를 고려해 crop한 이미지의 사이즈를 조정
    if bg_w <= int(x2-x1) or bg_h<=int(y2-y1) :
      obj_crop = cv2.resize(obj[y1:y2,x1:x2], (int((x2-x1)*ratio), int((y2-y1)*ratio))) # *ratio
    else :
      # print("no ratio")
      obj_crop = cv2.resize(obj[y1:y2,x1:x2], (int((x2-x1)), int((y2-y1)))) # 
    crop_h, crop_w, _ = obj_crop.shape


    bg_crop_x, bg_crop_y = np.random.randint(bg_w - crop_w), np.random.randint(bg_h - crop_h)

    copy_paste = bg.copy()
    copy_paste[bg_crop_y:bg_crop_y+crop_h, bg_crop_x:bg_crop_x+crop_w] = obj_crop

    copy_paste, bg = self.transform(copy_paste), self.transform(bg)

    return copy_paste, bg 
    

  def __len__(self):
    return len(self.bg_imgs)


  def get_transform(self):

    return T.Compose([
                      T.ToPILImage(),
                      T.Resize([self.load_size, self.load_size]), 
                      T.ToTensor(), 
                      T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
