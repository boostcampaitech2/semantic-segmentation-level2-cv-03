_base_=[
    '../_base_/base_dataset.py',
    '../_base_/base_runtime.py',
    '../_base_/base_schdule.py',
]


# python3 tools/model_converters/swin2mmseg.py https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth pretrain/swin_base_patch4_window12_384_22k.pth
pretrained = 'pretrain/swin_base_patch4_window12_384_22k.pth'

MODEL_NAME = 'OCR_swinB'

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    pretrained=pretrained,
    backbone=dict(
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False
        ),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[128, 256, 512, 1024],
            channels=sum([128, 256, 512, 1024]),
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=11,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[128, 256, 512, 1024],
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
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006*2,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=8e-7)

runner = dict(
    _delete_=True,
    type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(
    _delete_=True,
    by_epoch=True, interval=10)
evaluation = dict(
    _delete_=True,
    interval=1, metric='mIoU',save_best='mIoU')
# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=4)

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