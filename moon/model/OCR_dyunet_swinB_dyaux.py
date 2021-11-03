_base_=['OCR_swinB.py']

MODEL_NAME = 'OCR_dyunet_swinB_dyaux'
work_dir = f'./work_dirs/{MODEL_NAME}'

norm_cfg = dict(type='SyncBN', requires_grad=True)
model=dict(
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=128, # 상황봐서 조절
        num_outs=4
    ),
    decode_head=[
        dict(
            type='CustomDyUnetHead',
            scale=64,
            in_channels=[128]*4,
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
            in_channels=[128]*4,
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            channels=512,
            ocr_channels=256,
            dropout_ratio=-1,
            num_classes=11,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    auxiliary_head = dict(
            type='CustomDyUnetHead',
            scale=32,
            num_blocks=4,
            in_channels=[128]*4,
            in_index=(0,1,2,3),
            input_transform='multiple_select',
            channels=44,
            dropout_ratio=0.1,
            num_classes=11,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
)

data = dict(samples_per_gpu=int(4/1))
optimizer = dict(lr=0.00006/1)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))


log_config = dict(
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
         dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='p-stage_semantic_segmentation',
                name=MODEL_NAME,
                #entity='boostcampaitech2-object-detection-level2-cv-03'
                ))
    ])