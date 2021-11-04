import os
import json
from tqdm import tqdm


base = '/opt/ml/segmentation/input/data'
dst = '/opt/ml/segmentation/input/mmseg_general/'
train_imgs = json.load(open('/opt/ml/segmentation/input/data/train.json', 'r'))['images']
val_imgs = json.load(open('/opt/ml/segmentation/input/data/val.json', 'r'))['images']
test_imgs = json.load(open('/opt/ml/segmentation/input/data/test.json', 'r'))['images']

if not os.path.isdir(dst):
    # os.makedirs(dst)
    os.makedirs(os.path.join(dst, 'images', 'training'))
    os.makedirs(os.path.join(dst, 'images', 'validation'))
    os.makedirs(os.path.join(dst, 'test'))

for image in tqdm(train_imgs):
    f_name = os.path.join(dst, 'images', 'training', f'{image["id"]:04}.jpg')
    os.system(f'cp {os.path.join(base, image["file_name"])} {f_name}')
    # print(f'cp {os.path.join(base, image["file_name"])} {f_name}')
    # exit()

for image in tqdm(val_imgs):
    f_name = os.path.join(dst, 'images', 'validation', f'{image["id"]:04}.jpg')
    os.system(f'cp {os.path.join(base, image["file_name"])} {f_name}')

for image in tqdm(test_imgs):
    origin_name = image['file_name'].replace('/', '+')
    f_name = os.path.join(dst, 'test', f'{origin_name}')
    os.system(f'cp {os.path.join(base, image["file_name"])} {f_name}')