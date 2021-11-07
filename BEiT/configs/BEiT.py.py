norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='/opt/ml/segmentation/unilm/beit/semantic_segmentation/BEiT_Base_fixed_pt_real.pth',
    backbone=dict(
        type='BEiT',
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        img_size=512,
        init_values=0.1,
        drop_path_rate=0.1,
        out_indices=[3, 5, 7, 11]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(128, 128)))
dataset_type = 'CustomDataset'
data_root = '/opt/ml/segmentation/unilm/beit/mmseg_5fold_ver2_resam/fold2/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
classes = [
    'Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
    'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
]
multi_scale = [(512, 512), (544, 544), (576, 576), (608, 608), (640, 640),
               (672, 672), (704, 704), (736, 736), (768, 768), (800, 800),
               (832, 832), (864, 864), (896, 896), (928, 928), (960, 960),
               (992, 992), (1024, 1024)]
crop_size = (512, 512)
alb_transform = [
    dict(type='HorizontalFlip', p=0.5),
    dict(type='VerticalFlip', p=0.25),
    dict(type='ColorJitter', p=0.5),
    dict(type='RandomGridShuffle', p=0.5)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='Resize',
        img_scale=[(512, 512), (544, 544), (576, 576), (608, 608), (640, 640),
                   (672, 672), (704, 704), (736, 736), (768, 768), (800, 800),
                   (832, 832), (864, 864), (896, 896), (928, 928), (960, 960),
                   (992, 992), (1024, 1024)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.001),
    # dict(
    #     type='AugMix',
    #     json='/opt/ml/unilm/beit/semantic_segmentation/pseudo_dataset_v0.json',
    #     image_root='/opt/ml/segmentation/input/data'),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(
        type='Albu',
        transforms=[
            dict(type='HorizontalFlip', p=0.5),
            dict(type='VerticalFlip', p=0.25),
            dict(type='ColorJitter', p=0.5),
            dict(type='RandomGridShuffle', p=0.5)
        ]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(512, 512)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(512, 512)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='CustomDataset',
        data_root='/opt/ml/segmentation/unilm/beit/mmseg_5fold_ver2_resam/fold2/',
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize',
                img_scale=[(512, 512), (544, 544), (576, 576), (608, 608),
                           (640, 640), (672, 672), (704, 704), (736, 736),
                           (768, 768), (800, 800), (832, 832), (864, 864),
                           (896, 896), (928, 928), (960, 960), (992, 992),
                           (1024, 1024)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip', prob=0.001),
            # dict(
            #     type='AugMix',
            #     json=
            #     '/opt/ml/unilm/beit/semantic_segmentation/pseudo_dataset_v0.json',
            #     image_root='/opt/ml/segmentation/input/data'),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(
                type='Albu',
                transforms=[
                    dict(type='HorizontalFlip', p=0.5),
                    dict(type='VerticalFlip', p=0.25),
                    dict(type='ColorJitter', p=0.5),
                    dict(type='RandomGridShuffle', p=0.5)
                ]),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        classes=[
            'Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ]),
    val=dict(
        type='CustomDataset',
        data_root='/opt/ml/segmentation/unilm/beit/mmseg_5fold_ver2_resam/fold2/',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(512, 512)],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=[
            'Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ]),
    test=dict(
        type='CustomDataset',
        data_root='/opt/ml/segmentation/unilm/beit/mmseg_5fold_ver2_resam/fold2/',
        img_dir='images/test',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(512, 512)],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=[
            'Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ]))
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='BEiT',
                name='BEiT_fold2',
                entity='boostcampaitech2-object-detection-level2-cv-03'))
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(interval=10)
evaluation = dict(metric='mIoU', save_best='mIoU')
lr = 0.001
optimizer = dict(
    type='AdamW',
    lr=3e-05,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=327,
    warmup_ratio=0.1,
    min_lr_ratio=1e-06)
total_epochs = 40
work_dir = '/opt/ml/segmentation/unilm/beit/semantic_segmentation/work_dirs/fold2'
gpu_ids = range(0, 1)