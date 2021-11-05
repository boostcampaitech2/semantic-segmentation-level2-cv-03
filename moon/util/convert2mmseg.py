from pycocotools.coco import COCO
import numpy as np
import cv2
import argparse
import os
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Create annotations mask based on .json and Move image for mmseg dataset structures')

    parser.add_argument('data', help='data name')
    parser.add_argument('--anns_file_path_root', help='annotations file path root')
    parser.add_argument('--new_dataset_path_root',help='new datast path root that will be created')
    parser.add_argument('--origin_dataset_path',help='path that origin images are saved')
    parser.add_argument('--move_only', dest='cvt', action='store_false')
    parser.set_defaults(cvt=True)
    args = parser.parse_args()

    return args


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"



def convert2mask(
            data,
            anns_file_path_root = './stratified_kfold',
            new_dataset_path_root = '../dataset/annotations'):
    """
    convert json annotation to image annotation .png.
        data : The name of annotation json
        anns_file_path_root : dir path that dat json is being saved.
        new_dataset_path : dir path that image annotaiton will be saved, new_dataset_path must be following mmsegmentation dataset structure.
    """
    print()
    print('start convert..')
    print(f"\t data :{data}")
    print(f"\t anns_file_path_root :{os.path.abspath(anns_file_path_root)}")
    print(f"\t new_dataset_path_root :{os.path.abspath(new_dataset_path_root)}")
    print()

    if not os.path.exists(new_dataset_path_root):
        print(f"{new_dataset_path_root} don't exists.")
        os.makedirs(new_dataset_path_root,exist_ok=True)
        print(f"\tCreate dir {os.path.abspath(new_dataset_path_root)}")
    

    anns_file_path = os.path.join(anns_file_path_root,f'{data}.json')
    if not os.path.exists(anns_file_path):
        raise Exception(f'{os.path.abspath(anns_file_path)} don\'t exists!!')

    new_dataset_path = os.path.join(new_dataset_path_root,data)

    if not os.path.exists(new_dataset_path):
        os.mkdir(new_dataset_path)

    coco = COCO(anns_file_path)
    image_id = coco.getImgIds()
    image_infos = coco.loadImgs(image_id)
    category_names = ['Backgroud',
    'General trash',
    'Paper',
    'Paper pack',
    'Metal',
    'Glass',
    'Plastic',
    'Styrofoam',
    'Plastic bag',
    'Battery',
    'Clothing']

    image_num = 0
    for image_info in image_infos:
        ann_ids = coco.getAnnIds(imgIds=image_info['id'])
        anns = coco.loadAnns(ann_ids)

        # Load the categories in a variable
        cat_ids = coco.getCatIds()
        cats = coco.loadCats(cat_ids)

        # masks : size가 (height x width)인 2D
        # 각각의 pixel 값에는 "category id" 할당
        # Background = 0
        masks = np.zeros((image_info["height"], image_info["width"]))
        # General trash = 1, ... , Cigarette = 10
        anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
    
        for i in range(len(anns)):
            className = get_classname(anns[i]['category_id'], cats)
            pixel_value = category_names.index(className)
            masks[coco.annToMask(anns[i]) == 1] = pixel_value
        masks = masks.astype(np.int8)

        file_name = os.path.join(new_dataset_path,f"{image_num:04d}.png")
        image_num +=1
        cv2.imwrite(file_name,masks)

    print()
    print('finish convert !!')
    print()


def move2image(
            data,
            anns_file_path_root = './stratified_kfold',
            origin_dataset_path = '/opt/ml/segmentation/input/data',
            new_dataset_path_root = '../dataset/images'):
    """
    Move origin image to dataset/images for mmsegmentation dataset structure
        data : The name of annotation json
        anns_file_path_root : dir path that annotation file is saved.
        origin_dataset_path : dir path that origin image is saved.
        new_dataset_path_root : dir path that image  will be saved, new_dataset_path must be following mmsegmentation dataset structure.
    """

    print()
    print('start move..')
    print(f"\t data :{data}")
    print(f"\t anns_file_path_root :{os.path.abspath(anns_file_path_root)}")
    print(f"\t origin_dataset_path :{os.path.abspath(origin_dataset_path)}")
    print(f"\t new_dataset_path_root :{os.path.abspath(new_dataset_path_root)}")
    print()

    if not os.path.exists(new_dataset_path_root):
        print(f"{new_dataset_path_root} don't exists.")
        os.makedirs(new_dataset_path_root,exist_ok=True)
        print(f"\tCreate dir {os.path.abspath(new_dataset_path_root)}")

    new_dataset_path = os.path.join(new_dataset_path_root,data)
    anns_file_path = os.path.join(anns_file_path_root,f'{data}.json')

    if not os.path.exists(anns_file_path):
        raise Exception(f'{os.path.abspath(anns_file_path)} don\'t exists!!')

    if not os.path.exists(new_dataset_path):
        os.mkdir(new_dataset_path)

    coco = COCO(anns_file_path)
    image_id = coco.getImgIds()

    image_infos = coco.loadImgs(image_id)

    image_num = 0

    for image_info in image_infos:
        src_file = os.path.join(origin_dataset_path,image_info['file_name'])
        dest_file = os.path.join(new_dataset_path,f"{image_num:04d}.jpg")
        image_num+=1
        shutil.copy(src_file,dest_file)
    
    print()
    print('finish move !!')
    print()

def main():
    args = parse_args()

    mask_arg = dict()
    move_arg = dict()
    if args.data is not None:
        mask_arg['data'] = args.data
        move_arg['data'] = args.data
    if args.anns_file_path_root is not None:
        mask_arg['anns_file_path_root'] = args.anns_file_path_root
        move_arg['anns_file_path_root'] = args.anns_file_path_root
    if args.new_dataset_path_root is not None:
        mask_arg['new_dataset_path_root'] = os.path.join(args.new_dataset_path_root,'annotations')
        move_arg['new_dataset_path_root'] = os.path.join(args.new_dataset_path_root,'images')
    if args.origin_dataset_path is not None:
        move_arg['origin_dataset_path'] = args.origin_dataset_path

    print()
    print()
    print('='*100)

    if args.cvt:
        convert2mask(**mask_arg)
    move2image(**move_arg)

    print()
    print()
    print('='*100)
    
if __name__=='__main__':
    main()