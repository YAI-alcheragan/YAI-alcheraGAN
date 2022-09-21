from IPython.core.display import Image
import cv2
import numpy as np
import random
import sklearn
from sklearn.cluster import KMeans
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import matplotlib.pyplot as plt

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_colors(img, number_of_colors):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape(img.shape[0]*img.shape[1], 3)

    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(img)
    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    # hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    return rgb_colors

def match_image_by_color(image, color, threshold = 60, number_of_colors = 10): 
    
    image_colors = get_colors(image, number_of_colors)
    selected_color = rgb2lab(np.uint8(np.asarray([[color]])))

    select_image = False
    for i in range(number_of_colors):
        curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
        diff = deltaE_cie76(selected_color, curr_color)
        if (diff < threshold):
            select_image = True
    
    return select_image


def selectiveSearch(bg_img, cropped_img, num, search_cnt=500):
    h, w, _ = bg_img.shape
    ch, cw, _ = cropped_img.shape
    i = 0
    found = False
    COLORS = {'GREEN': [0, 128, 0]}
    while i < search_cnt and not found:
        point = (random.randint(0, w - cw), random.randint(0, h - ch))
        # crop_location_img = bg_img[point[1]:point[1]+ch, point[0]:point[0]+cw]
        below_crop_location_img = bg_img[point[1]+ch-int(ch*0.2):point[1]+ch, point[0]:point[0]+cw]
        selected = match_image_by_color(below_crop_location_img, COLORS['GREEN'], 70, 3)
        print(below_crop_location_img.shape)

        print(i+1,"th image trial")
        if selected:
        # # 수정 전 (-)    
        #     bg_img[point[1]:point[1]+ch, point[0]:point[0]+cw] = cropped_img
        #     copy_paste_img = bg_img
        #     cv2.imwrite("copy_paste_img{}.jpg".format(num),copy_paste_img)
            found = True
        i+=1
        
# 수정 후 (+)    
    if not i < search_cnt :
        print("selectiveSearch failed")
    return point


    



if __name__ == "__main__":
    number_of_imgs = 10
    for num in range(number_of_imgs):
        bg_img = cv2.imread('bg1.jpg')
        cropped_img = cv2.imread('crop_img1.jpg')
        selectiveSearch(bg_img, cropped_img, num)
