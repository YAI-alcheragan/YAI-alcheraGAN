import os
# import pandas as pd
import json
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torchvision.transforms as T
import glob
import torch
import cv2

from colormatch_meth1_main import selectiveSearch

ImageFile.LOAD_TRUNCATED_IMAGES = True

class BlendingDataset(Dataset):
    def __init__(self, root, bg_folder, obj_folder, mode, load_size=0):
      # Assuming objimg.jpg, objimg.json in obj_folder
        self.root = root
        self.bg_folder = os.path.join(root, mode, bg_folder)
        self.obj_folder = os.path.join(root, mode, obj_folder)

        print("path debug", self.bg_folder, self.obj_folder)
        self.mode = mode
        self.load_size = load_size

        self.bg_imgs = glob.glob(self.bg_folder+"/*.jpg")
      ## 수정 전 (-)
        # self.obj_imgs = glob.glob(self.obj_folder+"/*.jpg")
      ## 수정 후 (+)
        self.obj_imgs = glob.glob(self.obj_folder+"/**/*.jpg", recursive=True)
        self.transform = self.get_transform()

    def show_tensor(self, tensor, save=False, index=0):
        x = tensor.permute((1, 2, 0))
        x = x.to('cpu').detach().numpy()
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        cv2.imshow("tensor", x)
        cv2.waitKey()

    # # 수정 후 (-)
    # def save_mask(self, src_img, obj_img, bbox):
    #     mask = np.zeros_like(src_img)
    #     mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255
    #     cv2.imwrite("./src.jpg", cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
    #     cv2.imwrite("./dst.jpg", cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB))
    #     cv2.imwrite("./mask.jpg", cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

    def __getitem__(self, idx):
        bg = np.array(Image.open(self.bg_imgs[idx]))
        bg_h, bg_w, _ = bg.shape

        print("debug : " , len(self.obj_imgs))
        random_idx = np.random.randint(len(self.obj_imgs))

        obj_path = self.obj_imgs[random_idx]
      ## 수정 전(-)  
        obj_label_path = obj_path.replace("jpg", "json")
      ## 수정 후(+)
        #parent_dir, img_name = os.path.split(obj_path)

        #grandparent_dir = os.path.split(parent_dir)[0].replace("images", "labels")
        #obj_label_path = os.path.join(grandparent_dir, img_name ).replace("jpg", "json")
        
        with open(obj_label_path, "r") as js_file:
            obj_label = json.load(js_file)
        points = obj_label["objects"][0][0]['points']
        x1, y1, x2, y2 = points[0]['x'], points[0]['y'], points[1]['x'], points[1]['y']
        obj = np.array(Image.open(obj_path))

        obj_h, obj_w, _ = obj.shape

      
        ratio = min(bg_w/obj_w, bg_h/obj_h)   
        if bg_w <= int(x2-x1) or bg_h<=int(y2-y1) :
            cropped_obj = Image.fromarray(obj[y1:y2,x1:x2])
            obj_crop = np.array(cropped_obj.resize((int((x2-x1)*ratio), int((y2-y1)*ratio)))) # *ratio
        else :
            cropped_obj = Image.fromarray(obj[y1:y2,x1:x2])
            obj_crop = np.array(cropped_obj.resize((int((x2-x1)), int((y2-y1)))))#
        crop_h, crop_w, _ = obj_crop.shape

      #####
      # 수정 전 (-)
        #bg_crop_x, bg_crop_y = np.random.randint(bg_w - crop_w), np.random.randint(bg_h - crop_h)
        #print(obj_path)
        #self.save_mask(src_img=bg, obj_img=obj, bbox=[x1, y1, x2, y2])
      # 수정 후 (+)  : gpgan input으로 넣어줄 마스크 추가 
        bg_crop_x, bg_crop_y = selectiveSearch(bg, obj_crop, 0, search_cnt=500) 
        mask = np.zeros_like(bg)
        #mask = np.zeros((bg.shape[0],bg.shape[1]))
        mask[bg_crop_y:bg_crop_y+crop_h, bg_crop_x:bg_crop_x+crop_w, :] = 255       
      #####

        copy_paste = bg.copy()
        copy_paste[bg_crop_y:bg_crop_y+crop_h, bg_crop_x:bg_crop_x+crop_w] = obj_crop
        # 기현
        obj, copy_paste, bg, mask = self.transform(obj), self.transform(copy_paste), self.transform(bg), self.transform(mask)

        # 기현
        return {"obj": obj, "bg" : bg, "cp": copy_paste , "mask": mask}  #3 numpy inputs needed to run_gpgan



    def __len__(self):
        return len(self.bg_imgs)

    def get_transform(self):
        return T.Compose([
            T.ToPILImage(),
            T.Resize([self.load_size, self.load_size]),
            T.ToTensor(),
          # 수정 (-) : run_gpgan.py에서 normalize
            #T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

if __name__ == "__main__":
    root = "./datasets/"
    bg_folder, obj_folder = "bg", "smoke"
    dataset = BlendingDataset(root= root, bg_folder=bg_folder, obj_folder=obj_folder, mode="train", load_size=128)
    dataset.__getitem__(100)



