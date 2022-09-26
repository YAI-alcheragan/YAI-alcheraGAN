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
import sys

from colormatch_meth1_main import selectiveSearch

ImageFile.LOAD_TRUNCATED_IMAGES = True

class BlendingDataset(Dataset):
    def __init__(self, root, bg_folder, obj_folder, mode, load_size=0):
      # Assuming objimg.jpg, objimg.json in obj_folder
        self.root = root
        self.bg_folder = os.path.join(root, mode, bg_folder)
        self.obj_folder = os.path.join(root, mode, obj_folder)

        print(f"DEBUG // Assuming bg_folder has many [no_fire.jpg]  obj_folder has many [smoke.jpg, label.json, mask.jpg]")
        print(f"DEBUG // Assuming bg_folder data structure[{self.bg_folder}]     ==   data structure[datasets/skipped]  ")
        print(f"DEBUG // Assuming obj_folder data structure[{self.obj_folder}]  ==  data structure[datasets/confirmed] ")
        print(f"DEBUG // Make sure you have folders -> bg_folder, obj_folder")
        print(f"\t-> current bg_folder = {self.bg_folder}\n\t-> current obj_folder = {self.obj_folder}")

        self.mode = mode
        self.load_size = load_size

        # Assuming  bg_folder = [skipped]  obj_folder = [confirmed]
        # bg_folder로 지정된 경로에서 glob로 모든 jpg 파일들을 읽어옵니다
        # obj_folder로 지정된 경로에서 glob로 모든 img.jpg, img.json, mask.jpg 들을 가져옵니다
        # self.bg_imgs = glob.glob(self.bg_folder+"/**/*.jpg", recursive = True)
        # self.obj_imgs = glob.glob(self.obj_folder+"/**/*.jpg", recursive=True)

        self.bg_imgs = sorted(glob.glob(self.bg_folder+"/*/images/cur/*.jpg"))
        self.obj_imgs = sorted(glob.glob(self.obj_folder+"/*/images/cur/*.jpg"))
        self.bg_imgs = self.obj_imgs
        self.label_paths = sorted(glob.glob(self.obj_folder+"/*/labels/*.json"))
        self.mask_img_paths = sorted(glob.glob(self.obj_folder+"/*/masks/*.jpg"))

        self.transform = self.get_transform()
        
        print(f"DEBUG // bg_imgs # : {len(self.bg_imgs)}  obj_imgs # : {len(self.obj_imgs)}")
        print("blendingDataset __init__  finished")


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

        random_idx = np.random.randint(len(self.obj_imgs))

        obj_path = self.obj_imgs[random_idx]
      ## 수정 전(-)  
        # obj_label_path = obj_path.replace("jpg", "json")
      ## 수정 후(+)

        # # # 알체라 데이터셋 기준으로 
        # # obj_path = confirmed/#/images/cur/abc.jpg
        # # obj_label_path = confirmed/#/labels/abc.json 
        # # obj_mask_path = confirmed/#/masks/abc.jpg
        # parent_dir, img_name = os.path.split(obj_path)
        # grandparent_dir, _ = os.path.split(parent_dir)  # confirmed/#/images
        # obj_label_path = os.path.join(grandparent_dir.replace("images","labels"), img_name ).replace("jpg", "json")
        # obj_mask_path = os.path.join(grandparent_dir.replace("labels", "masks"), img_name )
        obj_label_path= self.label_paths[random_idx]
        obj_mask_path = self.mask_img_paths[random_idx]

        # bbox 좌표 로드
        with open(obj_label_path, "r") as js_file:
            obj_label = json.load(js_file)
        points = obj_label["objects"][0][0]['points']
        x1, y1, x2, y2 = points[0]['x'], points[0]['y'], points[1]['x'], points[1]['y']
        
        
        obj = np.array(Image.open(obj_path))
        mask = np.array(Image.open(obj_mask_path))
        obj_h, obj_w, _ = obj.shape

        print(f"DEBUG //  bg.shape ={bg.shape}, obj.shape={obj.shape}, mask.shape={mask.shape}")

        # # 아마도 비율을 고려해서 crop 해주는 부분 코드이므로, mask 도 자르기
        ratio = min(bg_w/obj_w, bg_h/obj_h)   
        if bg_w <= int(x2-x1) or bg_h<=int(y2-y1) :
            cropped_obj = Image.fromarray(obj[y1:y2,x1:x2])
            obj_crop = np.array(cropped_obj.resize((int((x2-x1)*ratio), int((y2-y1)*ratio)))) # *ratio
            cropped_mask = Image.fromarray(mask[y1:y2,x1:x2])
            mask_crop = np.array(cropped_mask.resize((int((x2-x1)*ratio), int((y2-y1)*ratio)))) 
        else :
            cropped_obj = Image.fromarray(obj[y1:y2,x1:x2])
            obj_crop = np.array(cropped_obj.resize((int((x2-x1)), int((y2-y1)))))#
            cropped_mask = Image.fromarray(mask[y1:y2,x1:x2])
            mask_crop = np.array(cropped_mask.resize((int((x2-x1)), int((y2-y1)))))#         
        crop_h, crop_w, _ = obj_crop.shape

        debug_cropped_obj = np.zeros_like(obj)
        debug_cropped_obj[y1:y2,x1:x2] = np.array(cropped_obj)
        debug_cropped_mask = np.zeros_like(mask)
        debug_cropped_mask[y1:y2,x1:x2] = np.array(cropped_mask)


        # color matching 으로 bg 이미지 위에 croped_obj(smoke)를 copy_paste할 적절한 위치 좌표 받음 
        bg_crop_x, bg_crop_y = selectiveSearch(bg, obj_crop, 0, search_cnt=1) 

        # 기존에 있던 연기 마스크를 새로운 좌표로 옮겨줘야 함 
        mask_new = np.zeros_like(bg)
        mask_new[bg_crop_y:bg_crop_y+crop_h, bg_crop_x:bg_crop_x+crop_w, :] = mask_crop
        #mask[bg_crop_y:bg_crop_y+crop_h, bg_crop_x:bg_crop_x+crop_w, :] = 255     

        # 기존에 있던 연기 사진도 새로운 좌표로 옮겨줘야 하는데, 이 경우 그냥 bg 이미지에 먼저 copy_paste 해버리는걸로
        copy_paste = bg.copy()
        copy_paste[bg_crop_y:bg_crop_y+crop_h, bg_crop_x:bg_crop_x+crop_w] = obj_crop

        # copy_paste 미리 해도 문제가 없는 이유 : gpgan의 input은 이미 copy_paste된 이미지이기 때문에
        copy_paste_init = copy_paste * mask_new + bg*(1-mask_new)

        obj, copy_paste, bg, mask_new = self.transform(obj), self.transform(copy_paste_init), \
                                    self.transform(bg), self.transform(mask_new)

        
        return {"obj": obj, "bg" : bg, "cp": copy_paste , "mask": mask_new, \
                "mask_old" : self.transform(mask), "cropped_obj": self.transform(debug_cropped_obj), "cropped_mask" : self.transform(debug_cropped_mask),\
                "cp_old" : self.transform(copy_paste) } 




    def __len__(self):
        return len(self.bg_imgs)

    def get_transform(self):
        return T.Compose([
            T.ToPILImage(),
            T.Resize([self.load_size, self.load_size]),
            T.ToTensor(),
          # 수정 (-) : gpgan.py에서 normalize
            #T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

if __name__ == "__main__":
    root = "./datasets/"
    bg_folder, obj_folder = "bg", "smoke"
    dataset = BlendingDataset(root= root, bg_folder=bg_folder, obj_folder=obj_folder, mode="train", load_size=128)
    dataset.__getitem__(100)



