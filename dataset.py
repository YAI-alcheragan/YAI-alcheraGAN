import torch
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os

import numpy as np
from PIL import Image

    

class SmokeDataset(Dataset):
    def __init__(self, data_root, load_size=256) :
        
        self.data_root = data_root
        
        self.smoke_img_root = os.path.join(data_root, 'confirmed')
        self.nosmoke_img_root = os.path.join(data_root, 'skipped')
        
        self.smoke_img_ls = []
        for img_num in os.listdir(self.smoke_img_root):
            self.smoke_img_ls += [os.path.join(self.smoke_img_root, img_num, 'images/cur', img_name) 
                            for img_name in os.listdir(os.path.join(self.smoke_img_root,img_num,'images/cur'))]
        self.nosmoke_img_ls = []
        for img_num in os.listdir(self.nosmoke_img_root):
            self.nosmoke_img_ls += [os.path.join(self.nosmoke_img_root, img_num, 'images/cur', img_name) 
                            for img_name in os.listdir(os.path.join(self.nosmoke_img_root,img_num,'images/cur'))] 

        
        self.total_img_ls = self.smoke_img_ls+self.nosmoke_img_ls
        self.total_label_ls = [1]*len(self.smoke_img_ls) + [0]*len(self.nosmoke_img_ls)
        
        self.transform = transforms.Compose([transforms.Resize([load_size, load_size]),
                                              transforms.ToTensor(), 
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    def __getitem__(self, idx):
        
        img_path = self.total_img_ls[idx]
        img = Image.open(img_path).convert("RGB")
        img_label = self.total_label_ls[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, img_label
    
        
    def __len__(self):
        
        return len(self.total_img_ls)
