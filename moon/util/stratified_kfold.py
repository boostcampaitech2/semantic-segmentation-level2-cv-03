
#!pip install scipy
# !pip3 uninstall scikit-learn --yes
# !pip3 install scikit-learn==0.22
# !pip install iterative-stratification

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from operator import add
import numpy as np
import json
import os
from tqdm import tqdm
import copy
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Split train_all.json to stratified k-fold json')
    parser.add_argument('--root', help='path that stratified kfold json are saved')
    parser.add_argument('--n_splits',type=int, help='The number of k')
    parser.add_argument('--annos_path',help='path that train_all.json is saved')

    args = parser.parse_args()

    return args


def stratified_kfold(
    root='./stratified_kfold',
    n_splits=5,
    annos_path = '/opt/ml/segmentation/input/data/train_all.json'):
    '''
    - X = 각 image

    - Y = image에 존재하는 label

    - image id에 대응하는 annotation label값을 모두 Y에 저장
        - len(X)=len(Y)= 데이터 개수
        
    - skfold로 label을 균등하게 나눈 index값을 구함

    - index = image_id이므로 해당 image_id를 가진 image와 annotations들을 모아서 값을 저장
    '''
    print()
    print()
    print('='*100)
    print("Start stratified_kfold")
    print(f"\t root :{root}")
    print(f"\t n_splits :{n_splits}")
    print(f"\t annos_path :{annos_path}")
    print()

    if not os.path.exists(root):
        print(f'root dir don\'t exist, create {os.path.abspath(root)}.')
        os.mkdir(root)
    print()

    with open(annos_path, 'rt', encoding='UTF-8') as annotations:
            coco = json.load(annotations)
            info = coco['info']
            licenses = coco['licenses']
            images = coco['images']
            annotations = coco['annotations']
            categories = coco['categories']

    X = coco['images']
    Y = [ [0]*len(categories) for _ in range(len(images))]

    for anno in annotations:
        image_id = anno['image_id']
        Y[image_id][anno['category_id']-1]+=1

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1010)

    imgid2annos = [[] for _ in range(len(coco['images']))]
    for anno in annotations:
        imgid = anno['image_id']
        imgid2annos[imgid].append(anno)

    print('='*100)
    print("start split !!")

    for idx,(train_index, val_index) in tqdm(enumerate(mskf.split(X, Y))):
        cv_train_path = os.path.join(root,f'cv_train{idx+1}.json')
        cv_val_path = os.path.join(root,f'cv_val{idx+1}.json')
        cv_train = dict()
        cv_val = dict()

        # train
        new_img_id=0
        new_ann_id=0

        cv_train['info'] = coco['info']
        cv_train['licenses'] = coco['licenses']
        cv_train['categories'] = coco['categories']
        
        train_images=[]
        train_annos=[]
        for t_index in train_index:
            image_id = X[t_index]['id']

            train_images.append(copy.deepcopy(X[t_index]))
            train_images[-1]['id'] = new_img_id
            
            train_anno = copy.deepcopy(imgid2annos[image_id])
            for idx in range(len(train_anno)):
                train_anno[idx]['id'] = new_ann_id
                train_anno[idx]['image_id'] = new_img_id
                new_ann_id+=1
            
            train_annos += train_anno
            new_img_id+=1
        
        cv_train['images'] = train_images
        cv_train['annotations'] = train_annos
        
        with open(cv_train_path,'w') as f:
            json.dump(cv_train,f,indent=1)
        
        # validation
        new_img_id=0
        new_ann_id=0
        cv_val['info'] = coco['info']
        cv_val['licenses'] = coco['licenses']
        cv_val['categories'] = coco['categories']
        
        val_images=[]
        val_annos=[]
        for v_index in val_index:
            image_id = X[v_index]['id']

            val_images.append(copy.deepcopy(X[v_index]))
            val_images[-1]['id'] = new_img_id
            

            val_anno = copy.deepcopy(imgid2annos[image_id])
            for idx in range(len(val_anno)):
                val_anno[idx]['id'] = new_ann_id
                val_anno[idx]['image_id'] = new_img_id
                new_ann_id+=1
 
            val_annos += val_anno
            new_img_id+=1
        
        cv_val['images'] = val_images
        cv_val['annotations'] = val_annos
        
        with open(cv_val_path,'w') as f:
            json.dump(cv_val,f,indent=1)
    
    print('='*100)
    print('finish stratified kfold !!')
    print(f'file saved in { os.path.join( os.path.abspath(root),"cv_[train,val]{version_number}")}')
    print()
    print()
    

def main():
    args = parse_args()
    kfold_arg = dict()
    if args.root is not None:
        kfold_arg['root'] = args.root
    if args.n_splits is not None:
        kfold_arg['n_splits'] = args.n_splits
    if args.annos_path is not None:
        kfold_arg['annos_path'] = args.annos_path

    stratified_kfold(**kfold_arg)

if __name__=='__main__':
    main()