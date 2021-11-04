from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json

dataroot = '/opt/ml/segmentation/input/data'

all_paths = glob(os.path.join(dataroot, '**', '*.jpg'))
print(len(all_paths))

r = []
g = []
b = []

for path in tqdm(all_paths):
    img = np.array(Image.open(path))
    # print(img.shape)
    r.append(img[:,:,0])
    g.append(img[:,:,1])
    b.append(img[:,:,2])

print(len(r), len(g), len(b))
print(np.mean(r))
print(np.mean(g))
print(np.mean(b))

print(np.std(r))
print(np.std(g))
print(np.std(b))

# trains = json.load(open('/opt/ml/segmentation/input/data/train.json', 'r'))['images']
# print(len(trains))