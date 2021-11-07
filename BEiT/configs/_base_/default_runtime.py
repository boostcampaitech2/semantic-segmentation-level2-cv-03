# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='BEiT',
                name='BEiT_Large_Aug',
                entity='boostcampaitech2-object-detection-level2-cv-03')
        )
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(interval=10)
evaluation = dict(metric='mIoU', save_best='mIoU')