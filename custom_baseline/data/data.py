import torch
from torch.utils.data import Dataset
import numpy as np
from utils.utils import *
from pycocotools.coco import COCO
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloader(mode='train', num_workers=4, batch_size=16, root='/opt/ml/segmentation/input/data'):
    if mode == 'train':
        transform = A.Compose([ToTensorV2()])
    elif mode == 'val':
        transform = A.Compose([ToTensorV2()])
    else:
        transform = A.Compose([ToTensorV2()])
    
    dataset = WasteDataset(root, mode, transform)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         num_workers=num_workers)
    return loader
        
class WasteDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.root = root
        if self.mode == 'train':
            self.coco = COCO(os.path.join(self.root, 'train.json'))
        elif self.mode == 'val':
            self.coco = COCO(os.path.join(self.root, 'val.json'))
        else:
            self.coco = COCO(os.path.join(self.root, 'test.json'))
        
    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        images = cv2.imread(os.path.join(self.root, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx : len(idx['segmentation'][0]), reverse=False)
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)
                        
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return {'image': images, 'mask': masks, 'info': image_infos['file_name']}
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return {'image': images, 'info': image_infos['file_name']}
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())