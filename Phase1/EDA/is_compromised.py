import os
from PIL import Image
from tqdm import tqdm
from glob import glob
import numpy as np

train_paths = glob('/opt/ml/segmentation/input/mmseg_single/images/training/*')
val_paths = glob('/opt/ml/segmentation/input/mmseg_single/images/validation/*')

raw_imgs = []
for path in val_paths:
    raw_imgs.append(Image.open(path))

cnt = 0
len_val = len(raw_imgs)
for path in tqdm(train_paths):
    train_img = Image.open(path)
    if train_img in raw_imgs:
        cnt += 1
        print(f'{cnt}\t / {len_val}\t{cnt/len_val*100}%')
