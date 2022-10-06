import os
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

class BlendingDataset4(Dataset):
    def __init__(self, root, bg_folder, obj_folder, load_size=256, crop_size = 200):
        """
        :param root: Confirmed, Skipped 폴더가 있는 상위 폴더 경로
        :param bg_folder: Skipped 폴더의 이름
        :param obj_folder: Confirmed 폴더의 이름
        :param load_size: 결과이미지 1개당 plot할 이미지 크기 (총 16개 이미지가 concat되어 출력)
        :crop_size: 연기가 난 부분 (bbox centor 좌표를 중심으로 정사각형) crop_size 할 크기. 이는 gpgan에 넣어 image blending 진행됨
        """

        self.root = root
        self.bg_folder = os.path.join(root, bg_folder)
        self.obj_folder = os.path.join(root, obj_folder)

        self.bg_imgs = sorted(glob.glob(self.bg_folder+"/*/images/cur/*.jpg"))
        self.obj_imgs = sorted(glob.glob(self.obj_folder+"/*/images/cur/*.jpg"))
        self.label_paths = sorted(glob.glob(self.obj_folder+"/*/labels/*.json"))
        self.mask_img_paths = sorted(glob.glob(self.obj_folder+"/*/masks/*.jpg"))
        self.load_size = load_size
        self.gp_input_size = int(crop_size)
        assert(self.gp_input_size%2 == 0)
        self.transform = self.get_transform()
        self.not_resize_tf = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
              ])
        
        print(f"help  // Assuming [{bg_folder}] has [image.jpg],  [{obj_folder}] has [image.jpg, label.json, mask.jpg]")
        print(f"      // Data structure follows [skipped], [confirmed] of alchera Dataset YAI_RGB_2203_2206 subfolder")
        print(f"\t-> current bg_folder = {self.bg_folder}\n\t-> current obj_folder = {self.obj_folder}")
        print(f"blend_dataset __init__ finished : bg_imgs # : {len(self.bg_imgs)}\tobj_imgs # : {len(self.obj_imgs)},\tmasks # : {len(self.mask_img_paths)}")

    def __getitem__(self, idx):

        # Target image 불러오기
        bg = np.array(Image.open(self.bg_imgs[idx]))
        bg_h, bg_w, _ = bg.shape

        # Source image 불러오기 - crop 영역이 이미지(obj_h = 1080, obj_w = 1920) 밖을 벗어나지 않는 이미지만 선택
        half = round(self.gp_input_size/2)
        while True :
          random_idx = np.random.randint(len(self.obj_imgs))

          obj = np.array(Image.open(self.obj_imgs[random_idx]))
          obj_h, obj_w, _ = obj.shape
          obj_label_path= self.label_paths[random_idx]
          with open(obj_label_path, "r") as js_file:
            obj_label = json.load(js_file)

          points = obj_label["objects"][0][0]['points']
          x1, y1, x2, y2 = points[0]['x'], points[0]['y'], points[1]['x'], points[1]['y']
          x_centor1 = round((x1+x2)/2)
          y_centor1 = round((y1+y2)/2)      

          crop_error_flag = ((x_centor1 + half+1) >=obj_w ) | ((x_centor1 - half-1) <=0 ) |  ((y_centor1 + half+1) >=obj_h ) | ((y_centor1 - half-1) <=0 )
          if not crop_error_flag :
            break

        # Mask image 불러오기
        mask = np.array(Image.open(self.mask_img_paths[random_idx]))


        # bbox 영역을 잘라서 연기 사진과 연기 마스크를 얻고, 둘을 곱해서 연기 마스크 모양의 연기 사진을 얻음
        cropped_obj = obj[y1:y2,x1:x2]  
        cropped_mask = mask[y1:y2,x1:x2]
        mask_crop = cropped_mask.copy() 
        obj_crop= cropped_obj.copy()
        crop_h, crop_w, _ = obj_crop.shape   

        mask_binary = mask_crop.copy()
        mask_binary[mask_binary>0.5]=1
        mask_binary[mask_binary<=0.5]=0
        smoke_in_bboxsize = obj_crop * mask_binary


        # color matching 알고리즘으로 bg 이미지 위에 smoke 를 copy_paste할 적절한 위치 좌표 받음  
        # bg_crop_x, bg_crop_y = selectiveSearch(bg, obj_crop, 0, search_cnt=1) 
        
        # color matching 알고리즘 이슈 발생, 현재 crop이 가능한 랜덤 위치로 copy paste
        while True :
          bg_crop_x, bg_crop_y = (random.randint(0, bg_w- crop_w), random.randint(0, bg_h - crop_h))
          x_centor = bg_crop_x + round(crop_w/2)
          y_centor = bg_crop_y + round(crop_h/2)

          error_flag = ((x_centor + half+1) >=bg_w ) | ((x_centor - half-1) <=0 ) |  ((y_centor + half+1) >=bg_h ) | ((y_centor - half-1) <=0 )
          if not error_flag :
            break
        
        # 기존에 있던 연기 마스크, 연기 사진을 방금 할당받은 새로운 좌표로 옮겨줘야 함 
        # mask_new는 새로운 위치의 흰색 연기 mask (0과 1 사이의 값, not binary)
        # smoke_img_new 는 새로운 위치의 흰색 연기 사진
        mask_new = np.zeros_like(bg)
        mask_new[bg_crop_y:bg_crop_y+crop_h, bg_crop_x:bg_crop_x+crop_w, :] = mask_crop
        smoke_img_new = np.zeros_like(bg)
        smoke_img_new[bg_crop_y:bg_crop_y+crop_h, bg_crop_x:bg_crop_x+crop_w, :] = smoke_in_bboxsize

        # GPGAN output과 비교용으로 만드는 copy_paste_init는 smoke_img_new을 bg 이미지에 붙여넣기
        mask_bin = mask_new.copy()
        mask_bin[mask_bin>0.5]=1
        mask_bin[mask_bin<=0.5]=0
        copy_paste_init = smoke_img_new*mask_bin + bg* (1-mask_bin) 

        # 결과물 비교에 도움이 될 사진들  
        debug_cropped_obj = np.zeros_like(obj)
        debug_cropped_obj[y1:y2,x1:x2] = np.array(cropped_obj)
        debug_cropped_mask = np.zeros_like(mask)
        debug_cropped_mask[y1:y2,x1:x2] = np.array(cropped_mask)
        copy_paste = bg.copy()
        copy_paste[bg_crop_y:bg_crop_y+crop_h, bg_crop_x:bg_crop_x+crop_w, :] = cropped_obj.copy()


        obj_tf, copy_paste_tf, bg_tf, mask_new_tf = self.transform(obj), self.transform(copy_paste_init), \
                                    self.transform(bg), self.transform(mask_new)


        # gpgan() 함수에 넣을 연기 부근 영역 crop 진행, gpgan 통과후 다시 crop 했던 위치에 붙여넣는 과정은 inference.gpgan.py에 구현
        gpgan_x1, gpgan_x2 = x_centor - half, x_centor + half
        gpgan_y1, gpgan_y2 = y_centor - half, y_centor + half

        gpgan_obj = obj[y_centor1-half : y_centor1+half , x_centor1-half : x_centor1+half]
        gpgan_bg = bg[gpgan_y1 : gpgan_y2 , gpgan_x1 : gpgan_x2]
        gpgan_smoke_img = np.array(smoke_img_new[gpgan_y1 : gpgan_y2 , gpgan_x1 : gpgan_x2])
        gpgan_mask = np.array(mask_new[gpgan_y1 : gpgan_y2 , gpgan_x1 : gpgan_x2])
        gpgan_cp = copy_paste_init[gpgan_y1 : gpgan_y2 , gpgan_x1 : gpgan_x2]
        
        gpgan_bg_inverse = bg  # 256 256 사이즈를 gpgan 통과 후 다시 1920 1080 이미지에 붙이기 위해서 원본 bg 이미지 필요

        gpgan_mask_bin = gpgan_mask.copy()
        gpgan_mask_bin[gpgan_mask_bin>0.5]=1
        gpgan_mask_bin[gpgan_mask_bin<=0.5]=0
        gpgan_obg = gpgan_obj * (1-gpgan_mask_bin) + gpgan_smoke_img * gpgan_mask_bin

        print('-'*50,'\n')
        print("debugging  __getitem__ ")
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
            T.Resize([self.load_size, self.load_size]),
            T.ToTensor(),
            ])

