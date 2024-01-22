#Vincent Fanditama Wijaya

import cv2 
from matplotlib import pyplot as plt
import os

target_img = cv2.imread('Dataset/Object.jpg')
target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
arr_data_img = []

for i in os.listdir('Dataset/Data'):
    img_temp = cv2.imread('Dataset/Data/'+i)
    img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    img_temp= cv2.GaussianBlur(img_temp, (5,5),3)
    arr_data_img.append(img_temp)

SURF = cv2.xfeatures2d.SURF_create()


surf_target_keypoint, surf_target_descrip = SURF.detectAndCompute(target_img, None)


FLANN = cv2.FlannBasedMatcher(dict(algorithm=1), dict(checks=50))

arr_imgs_idx = 0
total_match = 0
arr_imgs_keypoint= None
final_match = None
all_mask =[]

for idx, i in enumerate(arr_data_img):
    surf_arr_img_kp, surf_arr_img_des = SURF.detectAndCompute(i, None)
    
    flann = FLANN.knnMatch(surf_target_descrip, surf_arr_img_des, 2)

    arr_imgs_mask = []
    count_match =0
    for j in range(0, len(flann)):
        arr_imgs_mask.append([0, 0])
    
    for j, (first_match, second_match) in enumerate (flann):
        if(second_match.distance * 0.7 > first_match.distance):
            arr_imgs_mask[j] = [1,0]
            count_match+=1
    all_mask.append(arr_imgs_mask)
    
    if count_match > total_match:
        
        arr_imgs_keypoint = surf_arr_img_kp
        final_match = flann
        arr_imgs_idx = idx
        total_match = count_match

result = cv2.drawMatchesKnn(
    target_img, surf_target_keypoint, arr_data_img[arr_imgs_idx], arr_imgs_keypoint, final_match, None, matchColor=[0, 255, 0], singlePointColor=[255, 0, 0], matchesMask= all_mask[arr_imgs_idx]
)

plt.imshow(result)
plt.title('Tolak Angin')
plt.xticks([])
plt.yticks([])
plt.show()



