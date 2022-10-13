import os
import random
import glob
import shutil
from tqdm import tqdm

def checkDir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
        os.mkdir(dir+"/smoke")
        os.mkdir(dir + "/bg")
        os.mkdir(dir + "/mask")

def makeTrainTestDataset(img_paths, dst_path, label="smoke"):
    for img_path in tqdm(img_paths):
        img_path = img_path.replace("\\", "/")
        paths = img_path.split("/")
        folder_num, img_name = paths[3], paths[-1]
        if label == "smoke":
            src_path = smoke_path
            label_path = '{}/{}/{}/{}'.format(src_path, folder_num, "labels", img_name.replace("jpg", "json"))
            shutil.copy2(img_path, os.path.join(dst_path, img_name))
            shutil.copy2(label_path, os.path.join(dst_path, img_name.replace("jpg", "json")))
        if label == "bg":
            # src_path = bg_path
            # label_path = '{}/{}/{}/{}'.format(src_path, folder_num, "labels", img_name.replace("jpg", "json"))
            shutil.copy2(img_path, os.path.join(dst_path, img_name))
            # shutil.copy2(label_path, os.path.join(dst_path, img_name.replace("jpg", "json")))
        if label == "mask":
            shutil.copy2(img_path, os.path.join(dst_path, img_name))


smoke_path = "./datasets/confirmed"
bg_path = "./datasets/skipped"
mask_path = "./datasets/mask"

train_path = "./datasets/train"
test_path = "./datasets/test"
checkDir(train_path)
checkDir(test_path)
split_ratio = 0.9

smoke_img_paths = glob.glob(smoke_path+"/*/images/cur/*.jpg")
bg_img_paths = glob.glob(bg_path+"/*/images/cur/*.jpg")
mask_img_paths = glob.glob(mask_path+"/*.jpg")

smoke_img_paths.sort()
mask_img_paths.sort()

print("total smoke imgs: ", len(smoke_img_paths))
print("total bg imgs: ", len(bg_img_paths))
print("total mask imgs: ", len(mask_img_paths))

'''split train and test'''
random.shuffle(smoke_img_paths)
random.shuffle(bg_img_paths)
train_smoke_data_num = round(len(smoke_img_paths) * split_ratio)
train_bg_data_num = round(len(bg_img_paths) * split_ratio)

train_smoke_img_paths = smoke_img_paths[:train_smoke_data_num]
test_smoke_img_paths = smoke_img_paths[train_smoke_data_num:]

train_bg_img_paths = bg_img_paths[:train_bg_data_num]
test_bg_img_paths = bg_img_paths[train_bg_data_num:]

# train_mask_img_paths = mask_img_paths[:train_bg_data_num]
test_mask_img_paths = mask_img_paths[train_bg_data_num:]

print(f"train smoke imgs: {len(train_smoke_img_paths)}, test smoke imgs: {len(test_smoke_img_paths)}")
print(f"train bg imgs: {len(train_bg_img_paths)}, test bg imgs: {len(test_bg_img_paths)}")
print(f"test mask imgs: {len(test_mask_img_paths)}")

'''move train data'''
makeTrainTestDataset(train_smoke_img_paths, dst_path=os.path.join(train_path, "smoke"))
makeTrainTestDataset(test_smoke_img_paths, dst_path=os.path.join(test_path, "smoke"))
makeTrainTestDataset(train_bg_img_paths, dst_path=os.path.join(train_path, "bg"), label="bg")
makeTrainTestDataset(test_bg_img_paths, dst_path=os.path.join(test_path, "bg"), label="bg")
makeTrainTestDataset(test_mask_img_paths, dst_path=os.path.join(test_path, "mask"))
