#dataset settings
dataset_type = 'CustomDataset'
data_root = '/opt/ml/segmentation/jsp/'
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
crop_size = (512, 512)

# multi_scale = [(x,x) for x in range(512, 1024+1, 32)]
alb_transform=[
    dict(type='RandomCrop', width=224,height=224),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
]
    #     dict(type="HorizontalFlip",p=0.5),
    # dict(type='OneOf',transforms=[
    #     dict(type='RandomRotate90',p=1.0),
    #     dict(type='Rotate',limit=30,p=1.0),
    # ],p=0.5),
    # dict(type='OneOf',transforms=[
    #     dict(type='ElasticTransform',p=1, alpha=40, sigma=40 * 0.05, alpha_affine=40 * 0.03),
    #     dict(type='GridDistortion',p=1.0),
    #     dict(type='OpticalDistortion',distort_limit=1, shift_limit=0.5, p=1)
    # ],p=0.6),

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip',prob=0.3),
    #dict(type='Resize', img_scale=multi_scale, multiscale_mode='value', keep_ratio=True),
    dict(type='Resize', img_scale=(512,512), multiscale_mode='value', keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

        # A.HorizontalFlip(p=0.5),
        # # A.VerticalFlip(p=0.2),
        # A.OneOf([
        #     A.RandomRotate90(p=1.0),
        #     A.Rotate(limit=30, p=1.0),
        # ], p=0.5),

        # # A.RandomGamma(p=0.3),
        # # A.RandomBrightness(p=0.5),
        # A.OneOf([
        #     A.ElasticTransform(p=1, alpha=40, sigma=40 * 0.05, alpha_affine=40 * 0.03),
        #     A.GridDistortion(p=1),
        #     A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        # ], p=0.6),
test_pipeline = [
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
data = dict(
    samples_per_gpu=2,
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
        pipeline=test_pipeline,
        classes=classes,),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='annotations/validation',
        pipeline=test_pipeline,
        classes=classes,))