log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),  
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='Pstage-Seg',
                name='r50-cascade-rcnn')
        )
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