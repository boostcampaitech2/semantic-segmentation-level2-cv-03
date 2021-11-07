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


import cv2

from backbone import beit
model_cfg = '/opt/ml/segmentation/unilm/beit/semantic_segmentation/work_dirs/fold2_pseudo/fold2.py'
ckpt = '/opt/ml/segmentation/unilm/beit/semantic_segmentation/work_dirs/fold2_pseudo/best_mIoU_epoch_21.pth'
cfg = Config.fromfile(model_cfg)
cfg.data.test.test_mode = True

imsize = 512
cfg.data.test.pipeline[1]['img_scale'] = (imsize,imsize)
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

# cfg.model.pretrained = None
cfg.model.train_cfg = None

print('build model')
model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
print('build model complete')

print('loads ckpts')
ckpt = load_checkpoint(model, ckpt, map_location='cpu')
print('loads ckpts complete')

model.CLASSES = ckpt['meta']['CLASSES']
model = MMDataParallel(model.cuda(), device_ids=[0])

print('inference starts')
output = single_gpu_test(model, test_loader)
print('inference ends')

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

sample_subm = pd.read_csv('/opt/ml/segmentation/unilm/beit/sample_submission.csv', index_col=None)
result_subm = pd.read_csv('/opt/ml/segmentation/unilm/beit/sample_submission_empty.csv', index_col=None)
preds = [sub_dict[imId].flatten() for imId in sample_subm['image_id']]

sample_subm['PredictionString'] = [' '.join([str(dot) for dot in mask]) for mask in preds]



sample_subm.to_csv(f"/opt/ml/segmentation/unilm/beit/semantic_segmentation/work_dirs/submission/beit_fold2_pseudo.csv", index=False)
