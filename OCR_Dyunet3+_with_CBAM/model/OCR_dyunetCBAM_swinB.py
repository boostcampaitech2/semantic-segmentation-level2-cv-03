_base_ = ['OCR_dyunet_swinB_dyaux.py']

MODEL_NAME = 'OCR_dyunetCBAM_swinB'
fold = '2'
work_dir = f'./work_dirs/{MODEL_NAME}_cv{fold}'
data_root = '/opt/ml/segmentation/moon/dataset/'

norm_cfg = dict(type='SyncBN', requires_grad=True)

multi_scale = [(x,x) for x in range(256, 768+1, 32)]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    

####################################################################################

model = dict(
    backbone=dict(pretrain_img_size=384),
    neck=dict(
        out_channels=64
    ),
    decode_head=[
        dict(
            type='CustomDyUnetCBAMHead',
            scale=64,
            in_channels=[64]*4,
            in_index=(0,1,2,3),
            input_transform='multiple_select',
            channels=44,
            dropout_ratio=0.1,
            num_classes=11,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.6)),
        dict(
            type='OCRHead',
            in_channels=[64]*4,
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            channels=256,
            ocr_channels=128,
            dropout_ratio=-1,
            num_classes=11,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    auxiliary_head=dict(
        type='CustomDyUnetCBAMHead',
        in_channels=[64]*4)
)




train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='SegCutMix',p=0.3,data_root=data_root,
                        img_dir=f'images/cv_train{fold}',
                        ann_dir=f'annotations/cv_train{fold}',
                        class_weight=[0,0,0, .4, .05, .1, .35, 0,0,0, .1],
                        #class_weight=[0, 0, 0, 0.3, 0.05, 0.1, 0.35, 0, 0, 0, 0.2])
                        min_pixel=100
                        ),
    dict(type='MyAlbu'),
    dict(type='Resize', img_scale=multi_scale, multiscale_mode='value', keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]


data = dict(
    train=dict(
        img_dir = f"images/cv_train{fold}",
        ann_dir = f"annotations/cv_train{fold}",
        pipeline=train_pipeline,
    ),
    val=dict(
        img_dir = f"images/cv_val{fold}",
        ann_dir = f"annotations/cv_val{fold}"),
    samples_per_gpu = 4)

optimizer = dict(lr=0.00006)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

runner = dict(max_epochs=45)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
         dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='p-stage_semantic_segmentation',
                name=MODEL_NAME,
                ))
    ])