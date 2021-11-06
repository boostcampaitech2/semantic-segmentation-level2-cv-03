import albumentations as A
import matplotlib.pyplot as plt
import os
from pycocotools.coco import COCO
import cv2
import numpy as np

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

@PIPELINES.register_module()
class AugMix(object):
    def __init__(self, json, image_root, augmentation=True, max_obj=5, prob=0.5):
        """[summary]

        Args:
            json ([str]): pseudo json path
            image_root ([str]): original image root
            augmentation (bool, optional): apply augmentation to pasting object. Defaults to True.
            max_obj (int, optional): maximum number of obejct to paste (FAILED). Defaults to 5.
            prob (float, optional): percentage to apply. Defaults to 0.5.
        """
        self.augmentation = augmentation
        self.image_root = image_root
        self.root = COCO(json)
        self.max_obj = max_obj
        self.prob = prob
    def apply_augmentations(self, image, mask):
        transform = A.Compose([A.HorizontalFlip(p=0.5),
                               A.VerticalFlip(p=0.25),
                               A.ShiftScaleRotate(shift_limit=(-0.5, 0.5),
                                                  scale_limit=(-0.25, 0.2),
                                                  rotate_limit=(-60, 60),
                                                  border_mode=cv2.BORDER_CONSTANT, p=1.0)
                               ])
        
        transformed = transform(image=image, mask=mask)
        image, mask = transformed['image'], transformed['mask']
        return image, mask
    
    def get_classname(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id']==classID:
                return cats[i]['name']
        return "None"
    
    def __call__(self, results):
        if np.random.rand() < self.prob:
            return results
        # random select image and read image
        idx = np.random.randint(len(self.root.getImgIds()))
        if idx == 0:
            idx += 1
        image_id = self.root.getImgIds(imgIds=idx)
        image_infos = self.root.loadImgs(image_id)[0]
        
        image = cv2.imread(os.path.join(self.image_root, image_infos['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)           # unsigned int8
        
        # random select object
        idxs = self.root.getAnnIds(imgIds=image_infos['id'])
        if len(idxs) == 0:          # if no object return
            return results
        anns = self.root.loadAnns(idxs)
        obj_num = np.random.randint(len(idxs))        # max number of object
        
        if obj_num == len(anns):
            idxs = anns
        else:
            rnd_idxs = []
            for i in range(obj_num+1):
                while True:
                    rnd = np.random.randint(len(anns))
                    if idxs[rnd] not in rnd_idxs:
                        break
                rnd_idxs.append(idxs[rnd])
            idxs = self.root.loadAnns(rnd_idxs)
        
        # set read image and annotations
        cat_ids = self.root.getCatIds()
        cats = self.root.loadCats(cat_ids)
        
        mask = np.zeros((image_infos['height'], image_infos['width']))
        anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
        for i in range(len(anns)):
            className = self.get_classname(anns[i]['category_id'], cats)
            pix = category_names.index(className)
            mask[self.root.annToMask(anns[i]) == 1] = pix
        mask = mask.astype(np.int8)
        
        ori_img_w, ori_img_h = results['img'].shape[0], results['img'].shape[1]
        resize = A.Compose([A.Resize(width=ori_img_w, height=ori_img_h)])
        resized = resize(image=image, mask=mask)
        image, mask = resized['image'], resized['mask']
        
        if self.augmentation:
            image, mask = self.apply_augmentations(image, mask)
            
        # mask background to 0
        image[:][mask==0] = 0
        results['img'][image != 0] = image[image != 0]
        results['gt_semantic_seg'][mask != 0] = mask[mask != 0]
        return results


@PIPELINES.register_module()
class Albu:

    def __init__(self, transforms):
        self.transforms = transforms
        self.aug = Compose([self.albu_builder(t) for t in self.transforms])
        self.base = '/opt/ml/segmentation/mmsegmentation/submission/albu_vis'

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        It inherits some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    def __call__(self, results):
        img = results['img']
        mask = results['gt_semantic_seg']

        augmented = self.aug(image=img,mask=mask)

        results['img'] = augmented['image']
        results['gt_semantic_seg']= augmented['mask']

        return results