"""
Locate the codes below into [your conda env]/mmseg/datasets/pipelines/transforms.py
"""



import os
import matplotlib.pyplot as plt
try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


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
        # self._save_imgs(img, mask, results['filename'], 'origin')

        augmented = self.aug(image=img,mask=mask)

        results['img'] = augmented['image']
        results['gt_semantic_seg']= augmented['mask']
        # self._save_imgs(augmented['image'], augmented['mask'], results['filename'], 'transform')


        return results

    def _save_imgs(self, img, mask, fname, marker):
        fname = fname.split('/')[-1].split('.')[0]
        plt.imsave(os.path.join(self.base, f'{fname}_img_{marker}.jpg'), img)
        plt.imsave(os.path.join(self.base, f'{fname}_mask_{marker}.jpg'), mask)
        print(f'{fname} {marker}saved')