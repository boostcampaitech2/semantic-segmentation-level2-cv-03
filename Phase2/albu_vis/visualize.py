from PIL import Image
import os
from glob import glob
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
import pandas as pd
from matplotlib.lines import Line2D 
from matplotlib.colors import to_hex

csv_path = '/opt/ml/segmentation/mmsegmentation/submission/BEiT763+UNet768+SwinB743cv1-5.csv'
dataroot = '/opt/ml/segmentation/input/mmseg/test'
dstroot = f"/opt/ml/segmentation/mmsegmentation/submission/albu_vis/{csv_path.split('/')[-1].split('.')[0]}"
inference = pd.read_csv(csv_path)

if not os.path.isdir(dstroot):
    os.makedirs(dstroot)


classes = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic','Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
palette = [
    [0, 0, 0],
    [192, 0, 128], [0, 128, 192], [0, 128, 64],
    [128, 0, 0], [64, 0, 128], [64, 0, 192],
    [192, 128, 64], [192, 192, 128], [64, 64, 128], [128, 0, 192]
    ]

def rgb_to_hex(rgb):
    return '#' + ''.join(f'{c:02x}' for c in rgb)

cmap = {}
for cls, color in zip(classes, palette):
    cmap[cls] = rgb_to_hex(color)
handles = [Line2D([0], [0], marker='o', color=v, markerfacecolor=v, label=k, markersize=3) for k, v in cmap.items()]

images = inference['image_id'].tolist()
preds = inference['PredictionString'].tolist()


def str2img(cls):
    pass

row, col = 1, 3
plt.rcParams['axes.facecolor']='gray'
plt.rcParams['savefig.facecolor']='gray'
for img, pred in zip(tqdm(images), preds):
    board = np.zeros(shape=[256,256,3], dtype=np.uint8)
    for i, p in enumerate(pred.split()):
        board[i//256,i%256] = palette[int(p)]
    
    plt.clf()
    plt.subplot(row,col,1)
    plt.imshow(Image.open(os.path.join(dataroot, img.replace('/', '+'))))
    plt.axis('off')

    plt.subplot(row,col,2)
    plt.imshow(board)
    plt.axis('off')
    plt.legend(title='color', handles=handles, bbox_to_anchor=(2.0, 1.1), prop={'size': 6})
    plt.savefig(f"{dstroot}/{img.split('/')[-1]}", dpi=400)
