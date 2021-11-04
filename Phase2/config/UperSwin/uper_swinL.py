_base_ = [
    # '../../datasets/waste.py'
    '/opt/ml/segmentation/mmsegmentation/configs/_base_/datasets/waste.py'
]




###########################################################################
#Schedule
###########################################################################
lr = 1e-4  # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=7e-6)
# runtime settings
total_epochs = 60


###########################################################################
#Runtime
###########################################################################

expr_name = f'uperNet_swinB22k'
dist_params = dict(backend='nccl')

runner = dict(type='EpochBasedRunner', max_epochs=60)
checkpoint_config = dict(interval=9)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),  
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='Seg_psc',
                name=expr_name,
                entity='ark10806'
        ))
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
# evaluation = dict(save_best='val_mIoU', metric=['mIoU'])
evaluation = dict(metric='mIoU', pre_eval=True,save_best='mIoU')
# evaluation = dict(save_best='bbox_mAP', metric=['bbox'])
work_dir = './work_dirs/' + expr_name
gpu_ids = range(0, 1)


###########################################################################
#Model
###########################################################################
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)

# model = dict(
#     pretrained='pretrain/swin_base_patch4_window7_224.pth',
#     backbone=dict(
#         embed_dims=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
#     decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=150),
#     auxiliary_head=dict(in_channels=512, num_classes=150))

# _base_ = [
#     './upernet_swin_base_patch4_window12_512x512_160k_ade20k_'
#     'pretrain_384x384_1K.py'
# ]
# model = dict(pretrained='pretrain/swin_base_patch4_window12_384_22k.pth')
# pretrained='/opt/ml/segmentation/mmsegmentation/configs/_base_/models/UperSwin/pretrained/Converted_upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.pth',
emb = 192

model = dict(
    type='EncoderDecoder',
    pretrained='/opt/ml/segmentation/mmsegmentation/configs/_base_/models/UperSwin/pretrained/Converted_swin_large_patch4_window12_384_22kto1k.pth',
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=emb,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg,
        # init_cfg=dict(type='Pretrained', checkpoint='/opt/ml/segmentation/mmsegmentation/configs/swin/upernet_swin_base_patch4_window7_512x512.pth'),
        ),

    decode_head=dict(
        type='UPerHead',
        in_channels=[emb, emb*2, emb*4, emb*8],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))