import os
import json
from tqdm import tqdm


base = '/opt/ml/segmentation/input/data'
dst = '/opt/ml/segmentation/input/mmseg_general_distill/'
train = json.load(open('/opt/ml/segmentation/input/data/train.json', 'r'))
val = json.load(open('/opt/ml/segmentation/input/data/val.json', 'r'))
test = json.load(open('/opt/ml/segmentation/input/data/test.json', 'r'))

train_annots = train['annotations']
# val_annots = val['annotations']
# test_annots = test['annotations']

train_ann = {}
val_ann = {}
test_ann = {}

for ann in train_annots:
    if ann['image_id'] in train_ann.keys():
        train_ann[ann['image_id']].append(ann['category_id'])
    else:
        train_ann[ann['image_id']] = [ann['category_id']]

# for ann in val_annots:
#     if ann['image_id'] in train_ann.keys():
#         val_ann[ann['image_id']].append(ann['category_id'])
#     else:
#         val_ann[ann['image_id']] = [ann['category_id']]

# for ann in test_annots:
#     if ann['image_id'] in train_ann.keys():
#         test_ann[ann['image_id']].append(ann['category_id'])
#     else:
#         test_ann[ann['image_id']] = [ann['category_id']]


if not os.path.isdir(dst):
    os.makedirs(dst)
    os.makedirs(os.path.join(dst, 'images', 'training'))
    os.makedirs(os.path.join(dst, 'images', 'validation'))
    os.makedirs(os.path.join(dst, 'test'))


for image in tqdm(train['images']):
    if image['id'] in train_ann.keys():
        if 1 in train_ann[image['id']]:
            f_name = os.path.join(dst, 'images', 'training', f'{image["id"]:04}.jpg')
            os.system(f'cp {os.path.join(base, image["file_name"])} {f_name}')
        # print(f'cp {os.path.join(base, image["file_name"])} {f_name}')
        # exit()
    else:
        print('hi')


for image in tqdm(val['images']):
    f_name = os.path.join(dst, 'images', 'validation', f'{image["id"]:04}.jpg')
    os.system(f'cp {os.path.join(base, image["file_name"])} {f_name}')

for image in tqdm(test['images']):
    origin_name = image['file_name'].replace('/', '+')
    f_name = os.path.join(dst, 'test', f'{origin_name}')
    os.system(f'cp {os.path.join(base, image["file_name"])} {f_name}')