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

model_cfg = '/opt/ml/segmentation/mmsegmentation/configs/_base_/models/UperSwin/Customs/UperSwinB_kfold/ver4_final/fold_test.py'
ckpt = '/opt/ml/segmentation/mmsegmentation/configs/_base_/models/UperSwin/Customs/UperSwinB_kfold/ver4_final/work_dirs/final_fold3/best_mIoU_epoch_47.pth'
# ckpt = ''
# ckpt = ''

cfg = Config.fromfile(model_cfg)
cfg.data.test.test_mode = True

# imsize = 512
# cfg.data.test.pipeline[1]['img_scale'] = (imsize,imsize)
cfg.data.test.ann_dir = None

print('building dataset')
test_dataset = build_dataset(cfg.data.test)
print('finished building dataset')
test_loader = build_dataloader(
    test_dataset,
    samples_per_gpu=1,
    workers_per_gpu=1,
    dist=False,
    shuffle=False,
    drop_last=False
)

cfg.model.pretrained = None
cfg.model.train_cfg = None

model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
ckpt = load_checkpoint(model, ckpt, map_location='cpu')

model.CLASSES = ckpt['meta']['CLASSES']
model = MMDataParallel(model.cuda(), device_ids=[0])

output = single_gpu_test(model, test_loader)


# output_size = 512
output_size = 256
transform = A.Compose([A.Resize(output_size, output_size)])

sub_dict = {}
img_infos = test_dataset.img_infos
print('resize for submit')
for idx,  out in enumerate(tqdm(output)):
    image = np.zeros((1,1,1))
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

# submission.csv로 저장
# sample_subm.to_csv(f"/opt/ml/segmentation/mmsegmentation/submission/{model_cfg.split('/')[-1].split('.')[0]}_fold2_vanila717.csv", index=False)
sample_subm.to_csv(f"/opt/ml/segmentation/mmsegmentation/submission/refConfidence/{model_cfg.split('/')[-1].split('.')[0]}_final_fold3.csv", index=False)
# sample_subm.to_csv(f"/opt/ml/segmentation/mmsegmentation/submission/Similarity/script_default2.csv", index=False)
