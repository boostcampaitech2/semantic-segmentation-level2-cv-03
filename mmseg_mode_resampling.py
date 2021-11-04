from PIL import Image
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import copy
from collections import Counter
import cv2

ov_list = [1, 3, 4, 6, 11]

def get_mode(patch):
    cnt = Counter(patch)

    md = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
    if len(md) > 1:
        if (md[0][0] not in ov_list) and (md[1][0] in ov_list and md[1][1]>=2):
            return md[1][0]
    # md = max(cnt.items(), key=lambda x: x[1])[0]
    
    return md[0][0]


def mode_resampling(origin: np.array, kernel=3):
    mask = copy.deepcopy(origin)
    for i in range(0, 512-2):
        for j in range(0,512-2):
            curr = mask[i:i+kernel, j:j+kernel]
            mode_val = get_mode(origin[i:i+kernel, j:j+kernel].flatten())
            mask[i:i+kernel, j:j+kernel] = np.ones_like(curr) * mode_val
    
    return mask


# dataroot = '/opt/ml/segmentation/input/mmseg/annotations/training'
dataroot = '/opt/ml/segmentation/input/data/masks/origin'
mask_paths = glob(f'{dataroot}/*')
# dstroot = '/opt/ml/segmentation/input/mmseg/annotations/training_resam'
dstroot = '/opt/ml/segmentation/input/data/masks/mode_resampled_conditionalv2'

i = 0
for path in tqdm(mask_paths):
    mask = np.array(Image.open(path))
    resam = mode_resampling(mask)
    fname = path.split('/')[-1]
    cv2.imwrite(os.path.join(dstroot, fname), resam)
    i+= 1