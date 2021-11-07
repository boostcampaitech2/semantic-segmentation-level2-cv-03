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
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='inference test image')
    parser.add_argument('model_cfg', help='model config file name')
    parser.add_argument('model_path', help='model path that model config is saved')
    parser.add_argument('--submission_path',help='path that submission dir')
    parser.add_argument('--img_dir',help='path that test image is saved')
    parser.add_argument('--ms_tta', dest='ms_tta', action='store_true')
    parser.add_argument('--flip_tta', dest='flip_tta', action='store_true')
    parser.add_argument('--logit_save', dest='logit_save', action='store_true')

    parser.set_defaults(ms_tta=False)
    parser.set_defaults(flip_tta=False)
    parser.set_defaults(logit_save=False)
    
    args = parser.parse_args()

    return args

def inference(model_cfg,model_path,submission_path='./submission', img_dir='images/test', MS_TTA=False,FLIP_TTA=False,LOGIT_SAVE=False):
    BEST_CHECKPOINT = glob(os.path.join(model_path,'best_*'))
    #BEST_CHECKPOINT = ['/opt/ml/segmentation/moon/mmsegmentation/work_dirs/dyhead_swinB/epoch_50.pth']
    assert len(BEST_CHECKPOINT)==1
    BEST_CHECKPOINT = BEST_CHECKPOINT[0]

    cfg =Config.fromfile(os.path.join(model_path,model_cfg))
    cfg.data.test.test_mode = True

    size_min = 256
    size_max = 768
    if MS_TTA:
        multi_scale_list = [(x,x) for x in range(size_min, size_max+1, 32)]
        cfg.data.test.pipeline[1]['img_scale'] = multi_scale_list # Resize
    if FLIP_TTA:
        cfg.data.test.pipeline[1]['flip']=True

    cfg.data.test.img_dir = img_dir
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
        with open(os.path.join(model_path,'output.pth'),'wb') as f:
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
    sample = pd.read_csv( os.path.join(submission_path,'sample_submission.csv'), index_col=None )
    submission = pd.read_csv('/opt/ml/segmentation/baseline_code/submission/sample_submission.csv', index_col=None)
    for image_id in sample['image_id'].tolist():
        prediction_string = tmp_submission[tmp_submission['image_id']==image_id]['PredictionString'].iloc[0]
        submission = submission.append({"image_id" : image_id, "PredictionString" : prediction_string }, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv( os.path.join(submission_path,f"py_test_{model_cfg.split('.')[0]}.csv"), index=False)

def main():
    args = parse_args()
    infer_args = dict()
    infer_args['model_cfg'] = args.model_cfg
    infer_args['model_path'] = args.model_path
    infer_args['MS_TTA'] = args.ms_tta
    infer_args['FLIP_TTA'] = args.flip_tta
    infer_args['LOGIT_SAVE'] = args.logit_save

    if args.submission_path:
        infer_args['submission_path'] = args.submission_path
    if args.img_dir:
        infer_args['img_dir'] = args.img_dir
    
    inference(**infer_args)


if __name__=="__main__":
    main()