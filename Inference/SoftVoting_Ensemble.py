from random import sample
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                        wrap_fp16_model)
from mmcv.utils import DictAction
from mmcv import Config
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from glob import glob

import matplotlib.pyplot as plt
import albumentations as A
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

model_cfg = '/opt/ml/segmentation/mmsegmentation/configs/_base_/models/UperSwin/Customs/work_dirs/UperSwinB22k_1c_resamGENeral_noTTA_LB731/UperSwinB1c_ms_slide_resam_general_bettery.py'
# ckpt = '/opt/ml/segmentation/mmsegmentation/configs/_base_/models/UperSwin/Customs/work_dirs/UperSwinB22k_1a/best_mIoU_epoch_38.pth'
cfg = Config.fromfile(model_cfg)
cfg.data.test.test_mode = True

imsize = 512

print('building dataset')
test_dataset = build_dataset(cfg.data.test)


def multiple_ensemble(pkls):
    with open(pkls[0],"rb") as fr:
        ens = np.array(pickle.load(fr))
    
    for pkl_path in tqdm(pkls[1:]):
        ens += np.array(pickle.load(open(pkl_path,"rb")))
        
    return ens

def weighted_multiple_ensemble(pkls, weight):
    print('start weight')
    with open(pkls[0],"rb") as fr:
        ens = np.array(pickle.load(fr)) * weight[0]
    
    for idx, pkl_path in enumerate(tqdm(pkls[1:])):
        ens += np.array(pickle.load(open(pkl_path,"rb"))) * weight[idx+1]
        
    return ens


ens_paths = [
    '/opt/ml/segmentation/mmsegmentation/submission/refConfidence/final/tmp/OCR_dyunetCBAM_swinB_cv1-5_LB768.pkl',
    '/opt/ml/segmentation/mmsegmentation/submission/refConfidence/final/jsw_BEiT_loop3_LB771_minmax.pkl',
    '/opt/ml/segmentation/mmsegmentation/submission/refConfidence/final/tmp/SwinB_final_cv1-5_minmax.pkl',
]

output = multiple_ensemble(ens_paths)
# output = weighted_multiple_ensemble(ens_paths, weight=[0.6, 1.0, 1.0])
# output = multiple_ensemble(ens_paths, weight=[0.35, 0.4, 0.25])

# base = '/opt/ml/segmentation/mmsegmentation/submission/refConfidence/final'
# #######################################################################
# pkl_name = 'SwinB_final_cv1-5_new.pkl'
# with open(os.path.join(base, pkl_name), 'wb') as f:
#     pickle.dump(list(output), f)
#     print(f'{pkl_name} dumped!')
# #######################################################################


output = np.argmax(output, axis=1)


output_size = 256
transform = A.Compose([A.Resize(output_size, output_size)])

sub_dict = {}
img_infos = test_dataset.img_infos

print('resize for submit')
for idx,  out in enumerate(tqdm(output)):
    image = np.zeros((1,1,1))
    # image = np.zeros((1,output_size, output_size))
    transformed = transform(image=image, mask=out)
    mask = transformed['mask']

    mask = mask.reshape(-1, output_size*output_size).astype(int)
    sub_dict[img_infos[idx]['filename'].replace('+', '/')] = mask[0]
print('resize for submit complete')

# sample_submisson.csv 열기
sample_subm = pd.read_csv('/opt/ml/segmentation/baseline_code/submission/sample_submission.csv', index_col=None)
result_subm = pd.read_csv('/opt/ml/segmentation/baseline_code/submission/sample_submission_empty.csv', index_col=None)
# print(len(sample_subm['image_id']))
preds = [sub_dict[imId].flatten() for imId in sample_subm['image_id']]

sample_subm['PredictionString'] = [' '.join([str(dot) for dot in mask]) for mask in preds]
sample_subm.to_csv(f"/opt/ml/segmentation/mmsegmentation/submission/refConfidence/final/csv/final8_minmax.csv", index=False)