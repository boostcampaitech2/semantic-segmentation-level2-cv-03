import os
from glob import glob
from tqdm import tqdm
import cv2
import pickle
import numpy as np

print('='*100)
print("Start file load")

PATHS = [
    '/opt/ml/segmentation/moon/ensemble/OCR_dyunetCBAM_swinB_cv1-5.pth',
    '/opt/ml/segmentation/moon/mmsegmentation/work_dirs/pseudo_OCR_dyunetCBAM_swinB_cv1/output.pth',
    '/opt/ml/segmentation/moon/mmsegmentation/work_dirs/pseudo_OCR_dyunetCBAM_swinB_cv2/output.pth',
    # '/opt/ml/segmentation/moon/mmsegmentation/work_dirs/OCR_dyunetCBAM_swinB_cv1',
    # '/opt/ml/segmentation/moon/mmsegmentation/work_dirs/OCR_dyunetCBAM_swinB_cv2',
    # '/opt/ml/segmentation/moon/mmsegmentation/work_dirs/OCR_dyunetCBAM_swinB_cv3',
    # '/opt/ml/segmentation/moon/mmsegmentation/work_dirs/OCR_dyunetCBAM_swinB_cv4',
    # '/opt/ml/segmentation/moon/mmsegmentation/work_dirs/OCR_dyunetCBAM_swinB_cv5',
] # 모델 저장된 폴더

# LOGIT_PATHS = [glob(os.path.join(path,'output*')) for path in PATHS]

# assert len(PATHS)==len(LOGIT_PATHS)
#print(f"{len(LOGIT_PATHS)}'s outputs are loaded")
print(f"{len(PATHS)}'s outputs are loaded")

print('='*100)
print('Start Ensemble')

logits=None
for idx,logit_path in enumerate(tqdm(PATHS)):
    # assert len(logit_path)==1
    # logit_path = logit_path[0]

    with open(logit_path,'rb') as f:
        logit = np.array(pickle.load(f))

    if idx==0:
        logits=logit
    else:
        logits = logits + logit
        
logits/=len(PATHS)


output = [[] for _ in range(len(logits))]
for idx in tqdm(range(len(logits))):
    output[idx] = logits[idx].squeeze().argmax(axis=0)

print('Finish Ensemble')


pseudo_path = '/opt/ml/segmentation/moon/dataset/annotations/pseudo/'
DATAPATH = '/opt/ml/segmentation/moon/dataset/images/test'
file_names = sorted([ os.path.splitext(path)[0] for path in os.listdir(DATAPATH)])
assert len(file_names)==len(output)

print('='*100)
print('Start Pseudo labeling')
print(f'saved in {pseudo_path}')

for file_name,img in tqdm(zip(file_names,output)):
    file_path = os.path.join(pseudo_path,file_name+'.png')
    cv2.imwrite(file_path,img)

print('finish creating pseudo label!!')
