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
from mmcv import Config
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

##############################################################################################################################

MODEL = 'OCR_dyunetCBAM_swinB.py' # model config 경로
PATH = '/opt/ml/segmentation/moon/mmsegmentation/work_dirs/pseudo_OCR_dyunetCBAM_swinB_cv3' # 모델 저장된 폴더
SUBMISSION_PATH = '/opt/ml/segmentation/moon/submission/'
MS_TTA = False # multi scale TTA 여부
FLIP_TTA = False # flip TTA 여부
LOGIT_SAVE = False # logit 저장 여부

##############################################################################################################################
BEST_CHECKPOINT = glob(os.path.join(PATH,'best_*'))
#BEST_CHECKPOINT = ['/opt/ml/segmentation/moon/mmsegmentation/work_dirs/dyhead_swinB/epoch_50.pth']
assert len(BEST_CHECKPOINT)==1
BEST_CHECKPOINT = BEST_CHECKPOINT[0]

cfg =Config.fromfile(os.path.join(PATH,MODEL))
cfg.data.test.test_mode = True

size_min = 256
size_max = 768
if MS_TTA:
    multi_scale_list = [(x,x) for x in range(size_min, size_max+1, 32)]
    cfg.data.test.pipeline[1]['img_scale'] = multi_scale_list # Resize
if FLIP_TTA:
    cfg.data.test.pipeline[1]['flip']=True

cfg.data.test.img_dir = 'images/test'
cfg.data.test.ann_dir = None

test_dataset = build_dataset(cfg.data.test)
test_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

# build model
cfg.model.pretrained = None
cfg.model.train_cfg = None

checkpoint_path = BEST_CHECKPOINT

model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

model.CLASSES = checkpoint['meta']['CLASSES']
model.PALETTE = checkpoint['meta']['PALETTE']
model = MMDataParallel(model.cuda(), device_ids=[0])

output = single_gpu_test(model, test_loader) # 819x512x512 or 11x819x512x512

if LOGIT_SAVE:
    output2 = []
    for i in range(len(output)):
        output2.append(output[i].astype(np.float16))
    with open(os.path.join(PATH,'output.pth'),'wb') as f:
        pickle.dump(file=f,obj=output2)

if MS_TTA or FLIP_TTA:
    for idx in range(len(output)):
        output[idx] = output[idx].squeeze().argmax(axis=0)

size = 256
transform = A.Compose([A.Resize(size, size)])

file_name_list=[]
preds_array = np.empty((0,size*size),dtype=np.int64)
img_infos = test_dataset.img_infos

for idx,out in tqdm(enumerate(output)):
    image = np.zeros((1,size,size))
    transformed = transform(image=image,mask=out)
    mask = transformed['mask']
    
    mask = mask.reshape(-1,size*size).astype(int)
    preds_array = np.vstack((preds_array,mask))
    
    file_name_list.append(img_infos[idx]['filename'])


# sample_submisson.csv 열기
tmp_submission = pd.read_csv('/opt/ml/segmentation/baseline_code/submission/sample_submission.csv', index_col=None)

# PredictionString 대입
for file_name, string in zip(file_name_list, preds_array):
    file_name = '/'.join(file_name.split('+'))
    tmp_submission = tmp_submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                   ignore_index=True)

# submission.csv로 저장

# 순서도 같아야 채점이 되기 때문에 sample_submission이 필요
#sample = pd.read_csv( os.path.join(SUBMISSION_PATH,'sample_submission.csv'), index_col=None )
sample = pd.read_csv( os.path.join(SUBMISSION_PATH,'sample_submission.csv'), index_col=None )
submission = pd.read_csv('/opt/ml/segmentation/baseline_code/submission/sample_submission.csv', index_col=None)
for image_id in sample['image_id'].tolist():
    prediction_string = tmp_submission[tmp_submission['image_id']==image_id]['PredictionString'].iloc[0]
    submission = submission.append({"image_id" : image_id, "PredictionString" : prediction_string }, 
                                   ignore_index=True)

# submission.csv로 저장
submission.to_csv( os.path.join(SUBMISSION_PATH,f"py_test_{MODEL.split('.')[0]}.csv"), index=False)
    