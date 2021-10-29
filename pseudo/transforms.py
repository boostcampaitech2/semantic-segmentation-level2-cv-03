import albumentations as A
import matplotlib.pyplot as plt
import os

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
        # with open('/opt/ml/segmentation/input/data/train.json', 'r') as f:
        #     self.root = json.load(f)
    def apply_augmentations(self, image, mask):
        transform = A.Compose([A.HorizontalFlip(p=0.5),
                               A.VerticalFlip(p=0.25),
                            #    A.Rotate(p=0.25),
                            #    A.RandomBrightnessContrast(p=0.25)
                               A.ShiftScaleRotate(shift_limit=(-0.5, 0.5), scale_limit=(-0.25, 0.2), rotate_limit=(-60, 60), border_mode=cv2.BORDER_CONSTANT, p=1.0),
                                
                                #A.RandomBrightness(p=0.5),
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # cv2.imwrite(f'/opt/ml/unilm/beit/semantic_segmentation/debug/og/{image_id}.jpg', image)
        
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
        
        # self._save_imgs(results['img'],results['gt_semantic_seg'],'og', image_id)
        
        if self.augmentation:
            image, mask = self.apply_augmentations(image, mask)
        # mask background to 0
        image[:][mask==0] = 0
        # self._save_imgs(image/255.0, mask, 'pre_aug', image_id)
        results['img'][image != 0] = image[image != 0]
        results['gt_semantic_seg'][mask != 0] = mask[mask != 0]
        # self._save_imgs(results['img'],results['gt_semantic_seg'],'aug', image_id)
        # cv2.imwrite(f'/opt/ml/unilm/beit/semantic_segmentation/debug/aug/{image_id}.jpg', results['img'])
        return results
    
    def _save_imgs(self, img, mask, marker, image_id):
        plt.imsave(f'/opt/ml/unilm/beit/semantic_segmentation/debug/new_{marker}/{image_id}.jpg', img)
        plt.imsave(f'/opt/ml/unilm/beit/semantic_segmentation/debug/new_{marker}/{image_id}_mask.jpg', mask)
