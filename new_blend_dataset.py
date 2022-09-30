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


import random

from colormatch_meth1_main import selectiveSearch

ImageFile.LOAD_TRUNCATED_IMAGES = True

class BlendingDataset3(Dataset):
    def __init__(self, root, bg_folder, obj_folder, mode, load_size=0, gp_input_size = 512):
      # Assuming objimg.jpg, objimg.json in obj_folder
        self.root = root
        self.bg_folder = os.path.join(root, mode, bg_folder)
        self.obj_folder = os.path.join(root, mode, obj_folder)

        print(f"DEBUG // Assuming bg_folder has many [image.jpg]  obj_folder has many [image.jpg, label.json, mask.jpg]")
        print(f"DEBUG // Assuming [bg_folder], [obj_foler] data structure follows [skipped], [confirmed]")
        print(f"DEBUG // Make sure you have folders -> bg_folder, obj_folder")
        print(f"\t-> current bg_folder = {self.bg_folder}\n\t-> current obj_folder = {self.obj_folder}")

        self.mode = mode
        self.load_size = load_size
        self.gp_input_size = int(gp_input_size)
        assert(self.gp_input_size%2 == 0)

        # Assuming  bg_folder = [skipped]  obj_folder = [confirmed]
        # bg_folder로 지정된 경로에서 glob로 모든 jpg 파일들을 읽어옵니다
        # obj_folder로 지정된 경로에서 glob로 모든 img.jpg, img.json, mask.jpg 들을 가져옵니다
        # self.bg_imgs = glob.glob(self.bg_folder+"/**/*.jpg", recursive = True)
        # self.obj_imgs = glob.glob(self.obj_folder+"/**/*.jpg", recursive=True)

        self.bg_imgs = sorted(glob.glob(self.bg_folder+"/*/images/cur/*.jpg"))
        self.obj_imgs = sorted(glob.glob(self.obj_folder+"/*/images/cur/*.jpg"))
        self.label_paths = sorted(glob.glob(self.obj_folder+"/*/labels/*.json"))
        self.mask_img_paths = sorted(glob.glob(self.obj_folder+"/*/masks/*.jpg"))
        self.transform = self.get_transform()

        self.not_resize_tf = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
              ])
        
        print(f"DEBUG // bg_imgs # : {len(self.bg_imgs)}  obj_imgs # : {len(self.obj_imgs)}")
        print("blendingDataset __init__  finished")

    def show_tensor(self, tensor, save=False, index=0):
        x = tensor.permute((1, 2, 0))
        x = x.to('cpu').detach().numpy()
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        cv2.imshow("tensor", x)
        cv2.waitKey()

    def __getitem__(self, idx):
        bg = np.array(Image.open(self.bg_imgs[idx]))
        bg_h, bg_w, _ = bg.shape
        
        # gpgan에 넣어줄 input bg, obg, mask 사이즈를 여기서 정하면,
        # gpgan_input_size 대로 연기 부분만 crop될것
        half = round(self.gp_input_size/2)
        

        while True :

          random_idx = np.random.randint(len(self.obj_imgs))
          obj_label_path= self.label_paths[random_idx]
          with open(obj_label_path, "r") as js_file:
            obj_label = json.load(js_file)
          points = obj_label["objects"][0][0]['points']
          x1, y1, x2, y2 = points[0]['x'], points[0]['y'], points[1]['x'], points[1]['y']
          x_centor1 = round((x1+x2)/2)
          y_centor1 = round((y1+y2)/2)      

          flag = ((x_centor1 + half+1) >=1920 ) | ((x_centor1 - half-1) <=0 ) |  ((y_centor1 + half+1) >=1080 ) | ((y_centor1 - half-1) <=0 )
          #(1080,1920,3)
          if not flag :
            break

        # random_idx = np.random.randint(len(self.obj_imgs))


        # obj_label_path= self.label_paths[random_idx]
        # with open(obj_label_path, "r") as js_file:
        #   obj_label = json.load(js_file)
        # points = obj_label["objects"][0][0]['points']
        # x1, y1, x2, y2 = points[0]['x'], points[0]['y'], points[1]['x'], points[1]['y']

        mask = np.array(Image.open(self.mask_img_paths[random_idx]))
        obj = np.array(Image.open(self.obj_imgs[random_idx]))
        obj_h, obj_w, _ = obj.shape


        print('-'*50)
        print(f"DEBUG //  bg.shape ={bg.shape}, obj.shape={obj.shape}, mask.shape={mask.shape}") #(1080,1920,3)
        print(f"MIN values : [obj {obj.min()}], [bg {bg.min()}],[mask {mask.min()}]")
        print(f"MAX values : [obj {obj.max()}], [bg {bg.max()}],[mask {mask.max()}]")
        print('\n')

        # # # 아마도 비율을 고려해서 crop 해주는 부분 코드이므로, mask 도 자르기
        # ratio = min(bg_w/obj_w, bg_h/obj_h)   
        # if bg_w <= int(x2-x1) or bg_h<=int(y2-y1) :
        #     cropped_obj = Image.fromarray(obj[y1:y2,x1:x2])
        #     obj_crop = np.array(cropped_obj.resize((int((x2-x1)*ratio), int((y2-y1)*ratio)))) # *ratio
        #     cropped_mask = Image.fromarray(mask[y1:y2,x1:x2])
        #     mask_crop = np.array(cropped_mask.resize((int((x2-x1)*ratio), int((y2-y1)*ratio)))) 
        # else :
        #     cropped_obj = Image.fromarray(obj[y1:y2,x1:x2])
        #     obj_crop = np.array(cropped_obj.resize((int((x2-x1)), int((y2-y1)))))#
        #     cropped_mask = Image.fromarray(mask[y1:y2,x1:x2])
        #     mask_crop = np.array(cropped_mask.resize((int((x2-x1)), int((y2-y1)))))#   

        cropped_obj = obj[y1:y2,x1:x2]  
        obj_crop= cropped_obj.copy()
        cropped_mask = mask[y1:y2,x1:x2]

        mask_crop = cropped_mask.copy() 
        crop_h, crop_w, _ = obj_crop.shape   
        # mask_crop[mask_crop>0]=1

        #사각형 bbox 모양으로 crop된 연기 사진과 연기 마스크를 곱해서 
        #연기 마스크 모양의 연기 사진을 얻음
        mask_new_solved2 = mask_crop.copy()
        mask_new_solved2[mask_new_solved2>0.5]=1
        mask_new_solved2[mask_new_solved2<=0.5]=0
        only_smoke_in_bboxsize = obj_crop * mask_new_solved2

        # color matching 으로 bg 이미지 위에 croped_obj(smoke)를 copy_paste할 적절한 위치 좌표 받음 
        # bg_crop_x, bg_crop_y = selectiveSearch(bg, obj_crop, 0, search_cnt=1) 

        #gpgan에 넣을 crop 영역 256 256 위치 정하기
        while True :
          bg_crop_x, bg_crop_y = (random.randint(0, bg_w- crop_w), random.randint(0, bg_h - crop_h))
          x_centor = bg_crop_x + round(crop_w/2)
          y_centor = bg_crop_y + round(crop_h/2)

          flag = ((x_centor + half+1) >=1920 ) | ((x_centor - half-1) <=0 ) |  ((y_centor + half+1) >=1080 ) | ((y_centor - half-1) <=0 )
          if not flag :
            break
        
        # 기존에 있던 연기 마스크를 새로운 좌표로 옮겨줘야 함 
        # mask_new는 검은 바탕 위에 새로운 위치의 흰색 연기 mask (0과 1 사이의 값)
        mask_new = np.zeros_like(bg)
        mask_new[bg_crop_y:bg_crop_y+crop_h, bg_crop_x:bg_crop_x+crop_w, :] = mask_crop

        

        # 기존에 있던 연기 사진을 새로운 좌표로 옮겨줘야 함
        # smoke_img_new 는 검은 바탕 위에 흰색 연기 사진 (사각형 모양 아니고 segmentation 모양)
        smoke_img_new = np.zeros_like(bg)
        smoke_img_new[bg_crop_y:bg_crop_y+crop_h, bg_crop_x:bg_crop_x+crop_w, :] = only_smoke_in_bboxsize

        # copy_paste_init 은 연기 mask 모양으로 자른 연기 실제 사진을 bg 이미지에 붙여넣기
        mask_new_solved = mask_new.copy()
        mask_new_solved[mask_new_solved>0.5]=1
        mask_new_solved[mask_new_solved<=0.5]=0
        copy_paste_init = smoke_img_new*mask_new_solved + bg* (1-mask_new_solved) 

        # 시각화 이해에 도움이 될 사진들 
        debug_cropped_obj = np.zeros_like(obj)
        debug_cropped_obj[y1:y2,x1:x2] = np.array(cropped_obj)
        debug_cropped_mask = np.zeros_like(mask)
        debug_cropped_mask[y1:y2,x1:x2] = np.array(cropped_mask)
        copy_paste = bg.copy()
        copy_paste[bg_crop_y:bg_crop_y+crop_h, bg_crop_x:bg_crop_x+crop_w, :] = cropped_obj.copy()


        obj_tf, copy_paste_tf, bg_tf, mask_new_tf = self.transform(obj), self.transform(copy_paste_init), \
                                    self.transform(bg), self.transform(mask_new)


        gpgan_x1, gpgan_x2 = x_centor - half, x_centor + half
        gpgan_y1, gpgan_y2 = y_centor - half, y_centor + half
        print(mask_new.shape)

        # gpgan_obj = obj[gpgan_y1 : gpgan_y2 , gpgan_x1 : gpgan_x2]
        gpgan_obj = obj[y_centor1-half : y_centor1+half , x_centor1-half : x_centor1+half]
        gpgan_bg = bg[gpgan_y1 : gpgan_y2 , gpgan_x1 : gpgan_x2]
        gpgan_smoke_img = np.array(smoke_img_new[gpgan_y1 : gpgan_y2 , gpgan_x1 : gpgan_x2])
        gpgan_mask = np.array(mask_new[gpgan_y1 : gpgan_y2 , gpgan_x1 : gpgan_x2])
        gpgan_cp = copy_paste_init[gpgan_y1 : gpgan_y2 , gpgan_x1 : gpgan_x2]
        
        gpgan_bg_inverse = bg  # 256 256 사이즈를 gpgan 통과 후 다시 1920 1080 이미지에 붙이기 위해서 원본 bg 이미지 필요

        gpgan_mask_solved = gpgan_mask.copy()
        gpgan_mask_solved[gpgan_mask_solved>0.5]=1
        gpgan_mask_solved[gpgan_mask_solved<=0.5]=0
        gpgan_obg = gpgan_obj * (1-gpgan_mask_solved) + gpgan_smoke_img * gpgan_mask_solved


        
        

        print(f"[obj {obj.dtype}], [bg {bg.dtype}],[cp {copy_paste_init.dtype}] [mask {mask_new.dtype}]")
        print(f"MIN values : [obj {obj.min()}], [bg {bg.min()}],[cp {copy_paste_init.min()}] [mask {mask_new.min()}]")
        print(f"MAX values : [obj {obj.max()}], [bg {bg.max()}],[cp {copy_paste_init.max()}] [mask {mask_new.max()}]")
        print(f"[gpgan_obj {gpgan_obj.shape}], [gpgan_bg {gpgan_obj.shape}],[gpgan_mask {gpgan_mask.shape}] [gpgan_cp {gpgan_cp.shape}]")
        print(f"[gpgan_obj {type(gpgan_obj)}], [gpgan_bg {type(gpgan_bg)}],[gpgan_mask {type(gpgan_mask)}] [gpgan_cp {type(gpgan_cp)}]")
        print('-'*50,'\n')
        # sys.exit()
        return {"region" : (gpgan_x1,gpgan_x2, gpgan_y1,gpgan_y2),\
                "bg_final" : gpgan_bg_inverse, \
                "obj": obj_tf, "bg" : bg_tf, "cp": copy_paste_tf , "mask": mask_new_tf, \
                "gpgan_obj" : self.not_resize_tf(gpgan_obj), "gpgan_bg" : self.not_resize_tf(gpgan_bg), "gpgan_mask" : self.not_resize_tf(gpgan_mask),"gpgan_cp":self.not_resize_tf(gpgan_cp),\
                "mask_old" : self.transform(mask), "cropped_obj": self.transform(debug_cropped_obj), "cropped_mask" : self.transform(debug_cropped_mask),\
                "cp_old" : self.transform(copy_paste)  } 




    def __len__(self):
        return len(self.bg_imgs)

    def get_transform(self):
        return T.Compose([
            T.ToPILImage(),
            # T.ToTensor(),
            T.Resize([self.load_size, self.load_size]),
            T.ToTensor(),
          # 수정 (-) : gpgan.py에서 normalize
            #T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

