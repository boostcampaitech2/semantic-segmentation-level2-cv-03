from PIL import Image
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
from torch.autograd import Variable

import matplotlib.pyplot as plt
import albumentations as A
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from utils import synthetic
from convcrf import convcrf

model_cfg = '/opt/ml/segmentation/mmsegmentation/configs/_base_/models/UperSwin/Customs/work_dirs/UperSwinB22k_1c_resamGENeral_noTTA_LB731/UperSwinB1c_ms_slide_resam_general_bettery.py'
# ckpt = '/opt/ml/segmentation/mmsegmentation/configs/_base_/models/UperSwin/Customs/work_dirs/UperSwinB22k_1a/best_mIoU_epoch_38.pth'
cfg = Config.fromfile(model_cfg)
cfg.data.test.test_mode = True

imsize = 512

print('building dataset')
test_dataset = build_dataset(cfg.data.test)


def conv_crf(img: np.array, prob: torch.Tensor):
    '''
    img: (512 512 3)
    prob: (1 11 512 512) after softmax
    '''
    prob = torch.Tensor(prob).unsqueeze(0)
    num_classes = 11
    img_shape = img.shape[0:2]
    config = convcrf.default_conf
    config['filter_size'] = 5
    config['blur'] = 1
    config['pyinn'] = False
    img = img.transpose(2,0,1)
    img = img.reshape([1,3,512,512])
    image = Variable(torch.Tensor(img))
    image = image.cuda()
    unary = Variable(prob)
    unary = unary.cuda()
    gausscrf = convcrf.GaussCRF(conf=config, shape=img_shape, nclasses=num_classes, use_gpu=True)
    gausscrf.cuda()
    prediction = gausscrf.forward(unary=unary, img=image,num_iter=200)

    return prediction.data.cpu()


def multiple_ensemble(pkls):
    """
    Returns:
        pkls: [819, 11, 512, 512]
    """
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

def save_ensembled():
    base = '/opt/ml/segmentation/mmsegmentation/submission/refConfidence'
    pkl_name = 'SwinB_fold1-5.pkl'
    with open(os.path.join(base, pkl_name), 'wb') as f:
        pickle.dump(list(output), f)
        print(f'{pkl_name} dumped!')

ens_paths = [
    '/opt/ml/segmentation/mmsegmentation/submission/refConfidence/final/tmp/OCR_dyunetCBAM_swinB_cv1-5_LB768.pkl',
    '/opt/ml/segmentation/mmsegmentation/submission/refConfidence/final/jsw_BEiT_loop3_LB771.pkl',
    '/opt/ml/segmentation/mmsegmentation/submission/refConfidence/final/tmp/SwinB_final_cv1-5.pkl',
]

# output = multiple_ensemble(ens_paths)
output = weighted_multiple_ensemble(ens_paths, weight=[0.35, 0.4, 0.25])
# save_ensembled()

testroot = '/opt/ml/segmentation/input/data'
img_infos = test_dataset.img_infos
# Avg 조정: 0~1
for idx, (out, img_path) in enumerate(tqdm(zip(output/1.5, img_infos), desc='ConvCRF', total=len(output))):
    curr_img = np.array(Image.open(os.path.join(testroot, img_path['filename'].replace('+','/'))))
    output[idx] = conv_crf(curr_img, out)

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
sample_subm.to_csv(f"/opt/ml/segmentation/mmsegmentation/submission/refConfidence/final/final6_weightedCRF.csv", index=False)