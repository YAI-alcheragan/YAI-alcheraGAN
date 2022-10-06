import torch
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os

from PIL import Image

    

class MetricDataset(Dataset):
    """
    create dataset for evaluation metric
    """
    def __init__(self, data_root, load_size=256) :
        
        self.data_root = data_root
        
        self.img_ls = [os.path.join(data_root, img) for img in os.listdir(self.data_root)]
        
        self.transform = transforms.Compose([transforms.Resize([load_size, load_size]),
                                              transforms.ToTensor(), 
                                             ]) # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    def __getitem__(self, idx):
        
        img_path = self.img_ls[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img
    
        
    def __len__(self):
        
        return len(self.img_ls)
