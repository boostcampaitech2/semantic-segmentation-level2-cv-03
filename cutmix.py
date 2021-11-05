import json
import pandas as pd
import cv2
import os
import numpy as np
import math
import sys
import random
import xml.etree.ElementTree as etree

from PIL import Image
from PIL import ImageDraw
from itertools import combinations
from matplotlib.path import Path

np.set_printoptions(threshold=sys.maxsize) 

dataset_base = 'input/data'

with open('input/data/train_all.json') as json_file:
    train_json = json.load(json_file)
    
#class name    
classes = {"General_trash":0, "Paper":1, "Paper_pack":2, "Metal":3, "Glass":4, 
           "Plastic":5, "Styrofoam":6, "Plastic_bag":7, "Battery":8, "Clothing":9}

for i in classes.keys():
    os.makedirs(os.path.join(dataset_base,'train_collage',i),exist_ok=True)
    
real_json = {}
real_json['info'] = train_json['info']
real_json['licenses'] = train_json['licenses']
real_json['annotations'] = train_json['annotations']
real_json['categories'] = train_json['categories']
real_json['images'] =  train_json['images']

def get_segment_crop(img,tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]

def fill_mask(mask):

    nx, ny = 512, 512
    poly_verts = mask

    # (<0,0> is at the top left of the grid)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T

    path = Path(poly_verts)
    grid = path.contains_points(points)
    grid = grid.reshape((ny,nx))

    return grid

def turn_to_uint(a):
    mask = np.zeros((512, 512), dtype=np.uint8)
    for j in range(512):
        for k in range(512):
            if a[j][k]==False:
                mask[j][k] = 0
            else:
                mask[j][k] = 255
        
    
    return mask

def make_ann(ann,mask,category_id):
    for j in range(512):
        for k in range(512):
            if mask[j][k]==255:
                ann[j][k] = category_id
    return ann

def get_bg_ann(background_img):
    img_name = background_img[-8:-4]+'.png'
    res = cv2.imread(os.path.join('/opt/ml/segmentation/moon/dataset/annotations/train',img_name))
    # res = np.multiply(res, 20) # visualize위한 부분 실제로는 빼야됨

    return res[:,:,0]



# 마스크 영상을 이용한 영상 합성
##bg img 디렉토리에 100장 정도 넣어놓고 랜덤으로 뽑아서 하도록 해야함.......!!!바꾸자

pass_category=[1,4,5,6,10] # 합성에 사용하고 싶지 않은 catagory들
background_img = os.listdir('/opt/ml/segmentation/moon/background')

for i in range(len(background_img)):
    
    background_img[i] = '/opt/ml/segmentation/moon/background/'+background_img[i]

# ['/opt/ml/segmentation/moon/dataset/images/train/0007.jpg','/opt/ml/segmentation/moon/dataset/images/train/0147.jpg','/opt/ml/segmentation/moon/dataset/images/train/0156.jpg','/opt/ml/segmentation/moon/dataset/images/train/0133.jpg','/opt/ml/segmentation/moon/dataset/images/train/0131.jpg','/opt/ml/segmentation/moon/dataset/images/train/0030.jpg','/opt/ml/segmentation/moon/dataset/images/train/0086.jpg','/opt/ml/segmentation/moon/dataset/images/train/0138.jpg','/opt/ml/segmentation/moon/dataset/images/train/0664.jpg']
adding_point=[(300,10),(20,200),(0,10),(256,256),(10,300),(100,200)]


#annotations 겹치는 경우 어떻게 하는지, COCO에서는 다른그거에서는? 위에 있는거만 표시하는지??
##bg ann 가져오는거 만들어야함 Done
##하나의 이어져 있는 객체인지 확인해야함 Done

idx = 0
lala = 0
im_id = 5283




while True:
    if im_id == 8000:
        break
    for k in background_img:
        im_cnt=0
        bg_img = cv2.imread(k)
        ann = get_bg_ann(k)
        # cv2.imwrite(f"bg_test.png",ann)
        while True:
        
        # for idx in range(len(real_json['annotations'])):
            i = real_json['annotations'][idx]
            image_id = int(i['image_id'])
            category_id = i['category_id']
            img_path = real_json['images'][image_id]['file_name']
            area = i['area']
        
            if category_id not in pass_category:
                idx+=1
                continue

        
            img = cv2.imread(os.path.join(dataset_base,img_path))
            img_moved = np.zeros((512, 512,3), dtype=np.uint8)
            mask_moved = np.zeros((512, 512), dtype=np.uint8)
            mask_t = []
        
            #아래 이중 for문은 하나의 annotation가져오는것임
            for j in range(len(i['segmentation'])):
                for k in range(len(i['segmentation'][j])//2):
                    x = i['segmentation'][j][2*k]
                    y = i['segmentation'][j][2*k+1]
                    mask_t.append((x,y)) #좌표 collect
            #mask_t는 annotation 좌표 tuple로 저장한 list
            #mask : 좌표에 class값넣은것.
            mask = fill_mask(mask_t)
            mask = turn_to_uint(mask) 
        
            res, _ ,a,center = cv2.connectedComponentsWithStats(mask)
            x,y,w,h,_ = a[1] #object bbox
        
            # object가 하나로 이어져 있지 않거나 크기가 너무 작으면 패스
            if res>2 or area <25000 or area>1700000:
                idx+=1
                continue

            a = cv2.copyTo(img,mask) # RGB로 물체부분 추출하기
    
            #####boundary 넘어가는지 체크 근데 한바퀴만 돌 수 있게함
            l = idx%6
            new_x = adding_point[l][0]
            new_y = adding_point[l][1]
            cnt=0
            while new_x+h >= 512 or new_y+w>=512:
                cnt+=1
                l=(l+1)%6
                new_x = adding_point[l][0]
                new_y = adding_point[l][1]
                if cnt==5:
                    break
            if cnt==5:
                idx+=1
                continue
                
            # im_cnt+=1
            ##### a = (512,512,3) mask = (512,512)
            mask_moved[new_x:new_x+h, new_y:new_y+w] = mask[y:y+h,x:x+w]
            img_moved[new_x:new_x+h, new_y:new_y+w] = a[y:y+h,x:x+w]
            bg_img = cv2.copyTo(img_moved,mask_moved,bg_img) 
        
            # annotations추가하는부분
            ann = make_ann(ann,mask_moved,category_id)
    
            idx+=1
            cv2.imwrite(f"moon/dataset/images/train/{im_id}.jpg",bg_img)
            cv2.imwrite(f"moon/dataset/annotations/train/{im_id}.png",ann)
            im_id+=1
            if im_id%10==0:
                print(im_id)
            break
            # if im_cnt%2==0:  #im_cnt는 한 이미지에 몇개 합성할지 고르는것
            #     cv2.imwrite(f"synthesis/images/{im_id}.jpg",bg_img)
            #     cv2.imwrite(f"synthesis/annotations/{im_id}.png",ann)
            #     im_id+=1
            #     break


    