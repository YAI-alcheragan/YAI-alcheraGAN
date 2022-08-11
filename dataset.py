import numpy as nn
import glob

transforms_ = [
    #transforms.ToPILImage(),
    #transforms.Resize((img_size,img_size)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

def MaskingTransformV2(img, cood):

    # img와 bbox (x,y,w,h)를 받아서 해당 부분 mask 처리
    H,W = img.shape[-2:]
    mask = np.ones((H,W), dtype="uint8")
    
    x1 = round((cood[1]) * W)
    y1 = round((cood[2]) * H)
    w = round((cood[3]) * W)
    h = round((cood[4]) * H)
    x1 = round(x1-w/2)
    y1 = round(y1-h/2)
    x2= x1 +  w
    y2 = y1 + h

    mask[y1:y2,x1:x2] = 0

    return img, mask 

class bboxImageDataset(Dataset):
    def __init__(self, img_root, label_root, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(img_root + '/*.*'))     #jpg 파일들 이름 리스트
        self.files_B = sorted(glob.glob(label_root + '/*.*'))   #(annotation [c,x,y,h,w]) txt 파일들 이름 리스트


        ### txt 파일들(bboxes)을 하나의 ndarray(len(dataset), 5) 로 변환
        self.bboxes_arr = np.array([[0, 0, 0, 0, 0]])    # 첫줄 임시 생성
        for file_name in self.files_B:
            f = open(file_name, 'r')
            data = np.array([list(map(float,f.readline().split(' ')))])
            self.bboxes_arr = np.append(self.bboxes_arr, data, axis=0)
        f.close()
        self.bboxes_arr = np.delete(self.bboxes_arr, 0, 0)  # 첫줄 지우기


    def __getitem__(self, index):
        
        item_A = self.transform(PIL.Image.open(self.files_A[index % len(self.files_A)]))
        cood = self.bboxes_arr[index]

        item_A, mask = MaskingTransformV2(item_A, cood)
        item_B = item_A * mask


        return item_A, mask, item_B
        #return {'A': item_A, 'B': mask}
        # item_A : 원 이미지 , mask : 마스크 , item_B : 마스킹 된 이미지

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))