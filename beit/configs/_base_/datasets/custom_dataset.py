#dataset settings
dataset_type = 'CustomDataset'
data_root = '/opt/ml/segmentation/mmseg_input/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

classes = ['Backgroud',
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

multi_scale = [(x,x) for x in range(512, 1024+1, 32)]
crop_size=(512,512)
alb_transform=[
    dict(type='HorizontalFlip', p=0.5),
    dict(type='VerticalFlip', p=0.25),
    dict(type='ColorJitter', p=0.5),
    dict(type="RandomGridShuffle",p=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=multi_scale, multiscale_mode='value', keep_ratio=True),
    dict(type='RandomFlip', prob=0.001),
    dict(type='AugMix', json='/opt/ml/unilm/beit/semantic_segmentation/pseudo_dataset_v2.json',
         image_root='/opt/ml/segmentation/input/data'),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='Albu',transforms=alb_transform),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(512, 512)],
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        img_ratios=[1.0, 1.5, 2.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline,
        classes=classes,),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=val_pipeline,
        classes=classes,),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='annotations/validation',
        pipeline=test_pipeline,
        classes=classes,))