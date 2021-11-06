# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/datasets/custom_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/custom_schedule.py'
]
crop_size = (512, 512)

model = dict(
    backbone=dict(
        type='BEiT',
        img_size=512,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=0.1,
        drop_path_rate=0.1,
        out_indices=[3, 5, 7, 11],

    ),
    decode_head=dict(
        in_channels=[768, 768, 768, 768],
        num_classes=11,
        channels=768,
    ),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=11
    ), 
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(128,128))
)

optimizer = dict(_delete_=True, type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=327,
    warmup_ratio=0.1,
    min_lr_ratio=1e-06)
# By default, models are trained on 8 GPUs with 2 images per GPU

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(512, 512)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    val=dict(pipeline=test_pipeline))

# do not use mmdet version fp16
# fp16 = None
optimizer_config = dict(
    grad_clip=None
)